import gymnasium as gym
import miniworld
from miniworld.envs import OneRoom
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import RecordVideo
import os
import numpy as np
import wandb


wandb.init(
    project="miniworld-trial",
    monitor_gym=True,
    name="trial1",
    sync_tensorboard=True
)


class ShapedOneRoom(OneRoom):
    def _gen_world(self):
        super()._gen_world()

        pos_box = np.array([9, 0, 9])
        pos_agent = np.array([2, 0, 2])

        self.place_entity(self.box, self.rooms[0], pos=pos_box, dir=0)
        self.place_agent(self.rooms[0], pos=pos_agent, dir=0)

        assert not self.intersect(self.box, pos_box, self.box.radius)
        assert not self.intersect(self.agent, pos_agent, self.agent.radius)

    def _reward(self):
        if self.near(self.box):
            return 100
        x1, _, z1 = self.agent.pos
        x2, _, z2 = self.box.pos

        return -np.linalg.norm(np.array([x1, z1]) - np.array([x2, z2]))
    
    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        reward += self._reward()
        if self.near(self.box):
            termination = True

        return obs, reward, termination, truncation, info
    

gym.register("ShapedOneRoom-v0", entry_point=ShapedOneRoom)

def make_env():
    env = gym.make("ShapedOneRoom-v0", render_mode="rgb_array")
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
# env = VecVideoRecorder(
#     env,
#     f"policy_recordings/{1}",
#     record_video_trigger=lambda x: x % 10000 == 0,
#     video_length=200,
# )

eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=10000,
                             deterministic=True, render=False)
wandb_callback = WandbCallback(
        model_save_path=f"models/{1}",
        verbose=2,
    )


class VideoRecorderCallback(BaseCallback):
    """
    Records videos of the agent's performance periodically during training
    using RecordVideo wrapper
    """
    def __init__(self, env_fn, video_folder='policy_recordings', 
                 video_freq=10000, verbose=1):
        super().__init__(verbose)
        self.env_fn = env_fn  # function that returns an environment
        self.video_folder = video_folder
        self.video_freq = video_freq
        os.makedirs(video_folder, exist_ok=True)
        
    def _on_step(self):
        if self.n_calls % self.video_freq == 0:
            # Create a new environment for recording
            env = self.env_fn()
            
            # Get a unique video name based on the current training step
            video_subfolder = os.path.join(self.video_folder, f"step_{self.n_calls}")
            
            try:
                # Wrap the environment with RecordVideo
                env = RecordVideo(
                    env, 
                    video_folder=video_subfolder,
                    episode_trigger=lambda _: True  # Record every episode
                )
                
                # Run one episode
                obs, _ = env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    # Step the environment (recording happens automatically)
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                
                # Log reward
                self.logger.record('eval/video_reward', total_reward)
                if self.verbose > 0:
                    print(f"Video recorded at step {self.n_calls}, reward: {total_reward}")
            
            finally:
                # Close the environment safely
                try:
                    # Try to access the underlying environment
                    if hasattr(env, 'env'):
                        env.env.close()
                    else:
                        env.close()
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Error closing environment: {e}")
            
        return True


# Define a function to create an environment for recording
def make_recording_env():
    return gym.make("ShapedOneRoom-v0", render_mode="rgb_array")

video_callback = VideoRecorderCallback(
    env_fn=make_recording_env,
    video_folder="policy_recordings",
    video_freq=10000,  # Record every 20k steps
    verbose=1
)

model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{1}")
model.learn(
    total_timesteps=100000,
    callback=[eval_callback, wandb_callback, video_callback],
)

# Close training environment
env.close()

# Create a new environment for recording the rollout
video_folder = "policy_recordings/final"
os.makedirs(video_folder, exist_ok=True)

try:
    # Create and wrap environment
    record_env = gym.make("ShapedOneRoom-v0", render_mode="rgb_array")
    record_env = RecordVideo(record_env, video_folder)
    
    # Rollout and record the trained policy
    num_episodes = 3
    for episode in range(num_episodes):
        obs, _ = record_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = record_env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        print(f"Episode {episode+1}, steps: {steps}, reward: {total_reward}")
    
finally:
    # Close the recording environment safely
    try:
        # Try to access the underlying environment
        if hasattr(record_env, 'env'):
            record_env.env.close()
        else:
            record_env.close()
    except Exception as e:
        print(f"Warning: Error closing environment: {e}")

print(f"Rollout videos saved to: {video_folder}")
