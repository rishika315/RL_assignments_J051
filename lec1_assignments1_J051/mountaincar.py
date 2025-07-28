import gymnasium as gym
import time

env = gym.make("MountainCar-v0", render_mode="human")  # enable rendering
observation, info = env.reset(seed=42)

for _ in range(200):
    action = env.action_space.sample()  # take random action
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
