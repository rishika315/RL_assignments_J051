from __future__ import annotations
import os
import sys
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable

# Ensure this script can import from the current folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .gridworld import MDP, State, Action

# Simple alias for heuristic function
HeuristicFn = Optional[Callable[[State], float]]
 # Simple alias for heuristic function


@dataclass
class MCTSConfig:
    gamma: float = 0.95
    c_uct: float = 1.4
    rollouts_per_search: int = 500
    max_depth: int = 200


@dataclass
class Node:
    state: State
    parent: Optional[Tuple["Node", Action]] = None
    children: Dict[Action, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)


@dataclass
class MCTS:
    mdp: MDP
    config: MCTSConfig
    heuristic: HeuristicFn = None
    rng: Optional[random.Random] = field(default=None)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random(0)

    def uct_score(self, parent_visits: int, child: Node) -> float:
        if child.visits == 0:
            return float('inf')
        exploitation = child.q_value()
        exploration = self.config.c_uct * math.sqrt(math.log(parent_visits) / child.visits)
        return exploitation + exploration

    def select(self, node: Node) -> Node:
        while True:
            if self.mdp.is_terminal(node.state):
                return node
            actions = list(self.mdp.actions(node.state))
            if len(node.children) < len(actions):
                return node
            best_score = float('-inf')
            best_child = None
            for a, child in node.children.items():
                score = self.uct_score(node.visits, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                return node
            node = best_child

    def expand(self, node: Node) -> Node:
        actions = list(self.mdp.actions(node.state))
        unexpanded = [a for a in actions if a not in node.children]
        if not unexpanded:
            return node
        a = self.rng.choice(unexpanded)
        transitions = list(self.mdp.transitions(node.state, a))
        next_state = transitions[0].next_state if transitions else node.state
        child = Node(state=next_state, parent=(node, a))
        node.children[a] = child
        return child

    def rollout(self, state: State) -> float:
        total_reward = 0.0
        discount = 1.0
        depth = 0
        while not self.mdp.is_terminal(state) and depth < self.config.max_depth:
            actions = list(self.mdp.actions(state))
            if not actions:
                break
            a = self.rng.choice(actions)
            r_prob = self.rng.random()
            acc = 0.0
            chosen_next_state = state
            reward = 0.0
            for t in self.mdp.transitions(state, a):
                acc += t.probability
                if r_prob <= acc:
                    chosen_next_state = t.next_state
                    reward = t.reward
                    break
            total_reward += discount * reward
            discount *= self.config.gamma
            state = chosen_next_state
            depth += 1

        if self.heuristic and not self.mdp.is_terminal(state):
            total_reward += discount * self.heuristic(state)
        return total_reward

    def backprop(self, node: Node, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            if node.parent is None:
                break
            node, _ = node.parent
            reward *= self.config.gamma

    def search(self, root_state: State) -> Action:
        root = Node(state=root_state)

        for _ in range(self.config.rollouts_per_search):
            leaf = self.select(root)
            child = self.expand(leaf)
            reward = self.rollout(child.state)
            self.backprop(child, reward)

        best_action = None
        best_visits = -1
        for a, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = a
        if best_action is None:
            actions = list(self.mdp.actions(root_state))
            if not actions:
                raise RuntimeError("MCTS called on terminal state")
            best_action = actions[0]
        return best_action
