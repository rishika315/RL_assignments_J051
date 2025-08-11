from __future__ import annotations

from .gridworld import make_default_grid, sample_next_state_and_reward
from .rtdp import RTDP, RTDPConfig, LinearDecay
from .mcts import MCTS, MCTSConfig


def run_rtdp():
    print("Running RTDP...")
    env = make_default_grid()
    cfg = RTDPConfig(
        gamma=0.95,
        episodes=50,
        max_steps=1000,
        epsilon_schedule=LinearDecay(start=0.5, end=0.05, steps=50),
    )
    agent = RTDP(env, cfg)
    agent.run()

    # Run evaluation episodes
    print("\nRTDP Results:")
    for ep in range(20):
        s = env.initial_state()
        steps = 0
        total_reward = 0
        while not env.is_terminal(s) and steps < 1000:
            # Choose greedy action (epsilon = 0)
            a = agent.select_action(s, 0.0)
            s, r = sample_next_state_and_reward(env, s, a, agent.rng)
            total_reward += r
            steps += 1
        print(f"Episode {ep+1}: Steps to goal = {steps}, Total reward = {total_reward}")


def run_mcts():
    print("\nRunning MCTS...")
    env = make_default_grid()
    cfg = MCTSConfig(gamma=0.95, c_uct=1.4, rollouts_per_search=200, max_depth=200)
    agent = MCTS(env, cfg)

    for ep in range(20):
        s = env.initial_state()
        steps = 0
        total_reward = 0
        while not env.is_terminal(s) and steps < 1000:
            a = agent.search(s)
            s, r = sample_next_state_and_reward(env, s, a, agent.rng)
            total_reward += r
            steps += 1
        print(f"Episode {ep+1}: Steps to goal = {steps}, Total reward = {total_reward}")


if __name__ == "__main__":
    run_rtdp()
    run_mcts()
