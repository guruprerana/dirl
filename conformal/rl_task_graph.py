from typing import Any, Callable, Dict, List, Tuple
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
import wandb
from wandb.integration.sb3 import WandbCallback

from conformal.nonconformity_score_graph import NonConformityScoreGraph
from conformal.video_recorder_callback import VideoRecorderCallback


class RLTaskGraph(NonConformityScoreGraph):
    def __init__(
            self, 
            spec_graph: List[Dict[int, str]], 
            env_name: str,
        ):
        self.spec_graph = spec_graph
        adj_lists = [[v for v in edges.keys()] for edges in spec_graph]
        super().__init__(adj_lists)

        self.env_name = env_name
        self.path_policies: Dict[Tuple[int], Any] = dict()
        self.init_states: Dict[Tuple[int], List[Any]] = dict()
        self.init_states[(0,)] = None
    
    def train_all_paths(
            self, 
            wandb_project_name: str, 
            n_samples: int, 
            final_policy_recordings: int=3
        ):
        stack = [(0,)]

        while stack:
            path = stack.pop()
            for target_v in self.adj_lists[path[-1]]:
                target_path = path + (target_v,)
                self._train_edge(
                    target_path, 
                    wandb_project_name, 
                    n_samples,
                    final_policy_recordings
                )
                stack.append(target_path)

    def _train_edge(
            self, 
            path: List[int], 
            wandb_project_name: str, 
            n_samples: int,
            final_policy_recordings: int=3
        ):
        edge = (path[-2], path[-1])
        task_str = self.spec_graph[edge[0]][edge[1]]

        path_file_str = "-".join(str(i) for i in path)
        edge_task_name = f"path-{path_file_str}-{task_str}"
        wandb.init(
            project=wandb_project_name,
            monitor_gym=True,
            name=edge_task_name,
            sync_tensorboard=True,
        )

        edge_init_states = self.init_states[tuple(path[:-1])]
        def make_env():
            env = gym.make(
                self.env_name, 
                render_mode="rgb_array", 
                task_str=task_str,
                init_states=edge_init_states,
            )
            env = Monitor(env)
            return env
        
        def make_eval_env():
            env = gym.make(
                self.env_name, 
                render_mode="rgb_array", 
                task_str=task_str,
                init_states=edge_init_states,
            )
            return env
        
        env = DummyVecEnv([make_env])

        eval_callback = EvalCallback(
            env, 
            best_model_save_path=f"./logs/{wandb_project_name}/{edge_task_name}/best_models",
            log_path=f"./logs/{wandb_project_name}/{edge_task_name}/logs", 
            eval_freq=10000,                 
            deterministic=True, 
            render=False,
        )
        wandb_callback = WandbCallback(verbose=2)
        video_callback = VideoRecorderCallback(
            env_fn=make_eval_env,
            video_folder=f"./logs/{wandb_project_name}/{edge_task_name}/policy_recordings",
            video_freq=10000,
            verbose=1,
        )

        model = DQN(
            "CnnPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=f"./logs/{wandb_project_name}/{edge_task_name}/tensorboard",
        )
        model.learn(
            total_timesteps=100000,
            callback=[eval_callback, wandb_callback, video_callback],
        )

        env.close()
        self.path_policies[tuple(path)] = model

        # do n_samples rollouts to obtain starting state distribution for next vertex
        env = make_eval_env()
        next_init_states = []
        for episode in range(n_samples):
            obs, info = env.reset()
            env_state = info["env_state"]
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                env_state = info["env_state"]
                done = terminated or truncated

            next_init_states.append(env_state)

        self.init_states[tuple(path)] = next_init_states

        #### record final_policy_recordings
        videos_folder = f"./logs/{wandb_project_name}/{edge_task_name}/final_policy_recordings"
        env = RecordVideo(env, video_folder=videos_folder, episode_trigger=lambda _: True)

        for _ in range(final_policy_recordings):
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, info = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            wandb.log({"eval/final_policy_recordings_cumrew": total_reward})

        try:
            if hasattr(env, 'env'):
                env.env.close()
            else:
                env.close()
        except Exception as e:
            print(f"Warning: Error closing environment: {e}")

        wandb.finish()

    def load_edge_policy(
            self,
            path: Tuple[int],
            log_folder: str="./logs", 
            subfolder: str="riskyminiworldenv1",
        ):
        edge = (path[-2], path[-1])
        task_str = self.spec_graph[edge[0]][edge[1]]

        path_file_str = "-".join(str(i) for i in path)
        edge_task_name = f"path-{path_file_str}-{task_str}"
        model_file = f"{log_folder}/{subfolder}/{edge_task_name}/best_models/best_model.zip"
        self.path_policies[path] = DQN.load(model_file)

    def load_path_policies(
            self, 
            log_folder: str="./logs", 
            subfolder: str="riskyminiworldenv1"
        ):
        stack = [(0,)]

        while stack:
            path = stack.pop()
            for target_v in self.adj_lists[path[-1]]:
                target_path = path + (target_v,)
                self.load_edge_policy(target_path, log_folder, subfolder)
                stack.append(target_path)
    
    def sample(self, target_vertex, n_samples, path, path_samples):
        assert len(path_samples) == n_samples

        task_str = self.spec_graph[path[-1]][target_vertex]
        def make_eval_env():
            env = gym.make(
                self.env_name, 
                render_mode="rgb_array", 
                task_str=task_str,
            )
            return env
        
        env = make_eval_env()
        model = self.path_policies[tuple(path) + (target_vertex,)]

        next_path_samples = []
        losses = []
        
        for sample in path_samples:
            obs, info = env.reset(options={"state": sample})
            loss_eval = info["loss_eval"]
            env_state = info["env_state"]
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                loss_eval = info["loss_eval"]
                env_state = info["env_state"]
                done = terminated or truncated

            next_path_samples.append(env_state)
            losses.append(loss_eval)

        return next_path_samples, losses

            