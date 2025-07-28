import gymnasium as gym
import time

# Create the Acrobot environment with rendering
env = gym.make("Acrobot-v1", render_mode="human")
obs, info = env.reset(seed=42)

for step in range(500):
    action = env.action_space.sample()  # Random action: -1, 0, or +1 torque
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.02)

    if terminated or truncated:
        print(f"Terminated at step {step + 1}")
        obs, info = env.reset()

env.close()
