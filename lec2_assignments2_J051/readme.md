In this assignment, 
we implemented three Multi-Armed Bandit (MAB) strategies—**Exploration Only**, **Exploitation Only**, and **Upper Confidence Bound (UCB)**—each defining a different approach to selecting arms based on reward estimation. 
The **Exploration Only** strategy picks an arm entirely at random, ignoring any feedback or learning, making it ideal for baseline comparisons but poor for real-world optimization. 
The **Exploitation Only** strategy always selects the arm with the highest estimated reward, exploiting current knowledge but failing to explore other potentially better arms, which can lead to suboptimal outcomes if early estimates are misleading. 
Finally, the **UCB algorithm** strikes a balance between exploration and exploitation by considering both the estimated reward and the uncertainty (confidence bound) for each arm. 
It prioritizes arms with higher potential by boosting under-explored arms, ensuring a more informed decision-making process over time. This progression from random to greedy to balanced strategies reflects increasing intelligence and adaptability in decision-making under uncertainty.

