# visualize.py
import matplotlib.pyplot as plt

# You can copy these from your console output or log them during runs
rtdp_steps = [9, 11, 11, 12, 12, 13, 11, 14, 11, 11, 11, 20, 11, 9, 11, 11, 16, 11, 12, 10]
rtdp_rewards = [-8, -10, -10, -11, -11, -12, -10, -13, -10, -10, -10, -19, -10, -8, -10, -10, -15, -10, -11, -9]

mcts_steps = [36, 34, 16, 24, 17, 25, 90, 51, 27, 33, 38, 77, 13, 15, 18, 16, 17, 19, 13, 15]
mcts_rewards = [-35, -33, -15, -23, -16, -24, -89, -50, -26, -32, -37, -76, -12, -14, -17, -15, -16, -18, -12, -14]

episodes = list(range(1, len(rtdp_steps) + 1))

plt.figure(figsize=(10, 5))

# Steps-to-goal plot
plt.subplot(1, 2, 1)
plt.plot(episodes, rtdp_steps, marker='o', label='RTDP')
plt.plot(episodes, mcts_steps, marker='s', label='MCTS')
plt.xlabel('Episode')
plt.ylabel('Steps to Goal')
plt.title('Steps to Goal per Episode')
plt.legend()
plt.grid(True)

# Reward plot
plt.subplot(1, 2, 2)
plt.plot(episodes, rtdp_rewards, marker='o', label='RTDP')
plt.plot(episodes, mcts_rewards, marker='s', label='MCTS')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
