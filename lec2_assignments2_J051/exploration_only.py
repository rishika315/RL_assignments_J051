import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class ExplorationOnly(BaseMABAlgorithm):
    """
    Pure exploration algorithm - randomly selects arms
    """
    def __init__(self, n_arms: int, **kwargs):
        super().__init__(n_arms, **kwargs)
        
    def select_arm(self) -> int:
        """
        Selects an arm purely at random.

        Returns:
            int: Randomly selected arm index (0 to n_arms - 1)
        """
        return np.random.randint(0, self.n_arms)
