import numpy as np
from algorithms.base_algorithm import BaseMABAlgorithm

class UCB(BaseMABAlgorithm):
    """
    Upper Confidence Bound (UCB) algorithm
    Balances exploration and exploitation using confidence bounds
    """
    def __init__(self, n_arms: int, c: float = 2.0, **kwargs):
        super().__init__(n_arms, **kwargs)
        self.c = c  # Exploration parameter
        
    def select_arm(self) -> int:
        """
        Selects the arm with the highest UCB value.

        Returns:
            int: Selected arm index
        """
        # Step 1: Pull untried arms first
        unpulled_arms = np.where(self.pulls == 0)[0]
        if len(unpulled_arms) > 0:
            return int(unpulled_arms[0])  # Explore first unpulled arm

        # Step 2: Compute UCB values
        total_pulls = np.sum(self.pulls)
        confidence_bounds = self.c * np.sqrt(np.log(total_pulls) / self.pulls)
        ucb_values = self.estimates + confidence_bounds

        # Step 3: Select arm with highest UCB
        return int(np.argmax(ucb_values))
