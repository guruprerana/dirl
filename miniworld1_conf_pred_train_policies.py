import gymnasium as gym
import miniworld
from miniworld.envs import OneRoom
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
from gymnasium.wrappers import RecordVideo
import os
import numpy as np
import wandb


wandb.init(
    project="miniworld-trial",
    monitor_gym=True,
    name="trial1"
)


class ShapedOneRoom(OneRoom):
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

env = gym.make("ShapedOneRoom-v0", render_mode="rgb_array")
env = Monitor(env)
env = VecVideoRecorder(
    env,
    f"policy_recordings/{1}",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(
    total_timesteps=50000, 
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{1}",
        verbose=2,
    ),
)

# Close training environment
env.close()

# Create a new environment for recording the rollout
video_folder = "policy_recordings"
os.makedirs(video_folder, exist_ok=True)
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

# Close the recording environment
record_env.close()
print(f"Rollout videos saved to: {video_folder}")
