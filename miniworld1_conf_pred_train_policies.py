import gymnasium as gym
import miniworld
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo
import os

env = gym.make("MiniWorld-FourRooms-v0", render_mode="rgb_array")

model = DQN("CnnPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=10000, log_interval=4)

# Close training environment
env.close()

# Create a new environment for recording the rollout
video_folder = "policy_recordings"
os.makedirs(video_folder, exist_ok=True)
record_env = gym.make("MiniWorld-FourRooms-v0", render_mode="rgb_array")
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
