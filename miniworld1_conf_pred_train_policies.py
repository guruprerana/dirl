import gymnasium as gym
import miniworld

env = gym.make("MiniWorld-OneRoom-v0", render_mode="human")

observation, info = env.reset(seed=42)
print(observation, info)
env.close()
