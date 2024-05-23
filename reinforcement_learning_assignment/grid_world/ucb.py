import numpy as np

class UCB:
    """
    Upper Confidence Bound (UCB) algorithm for multi-armed bandit problems.

    Attributes:
        num_actions (int): Number of arms (actions) in the bandit problem.
        Q (numpy array): Array of size num_actions to store action-value estimates.
        N (numpy array): Array of size num_actions to store action counts.
        c (float): Exploration parameter for UCB calculation.
    """

    def __init__(self, num_actions, c=1.0):
        """
        Initializes the UCB agent.

        Args:
            num_actions (int): Number of arms (actions) in the bandit problem.
            c (float, optional): Exploration parameter for UCB. Defaults to 1.0.
        """
        self.num_actions = num_actions
        self.Q = np.zeros(num_actions)  # Action-value estimates
        self.N = np.zeros(num_actions)  # Action counts
        self.c = c  # Exploration parameter

    def select_action(self, t):
        """
        Selects action using UCB exploration-exploitation strategy.

        Args:
            t (int): Time step or iteration number.

        Returns:
            int: Index of the selected action.
        """
        # Calculate UCB values for all actions
        ucb_values = self.Q + self.c * np.sqrt(np.log(t + 1) / (self.N + 1e-5))
        # Select action with maximum UCB value
        return np.argmax(ucb_values)

    def update(self, action, reward):
        """
        Updates action-value estimate and action count based on received reward.

        Args:
            action (int): Index of the action taken.
            reward (float): Reward received for taking the action.
        """
        # Update action count
        self.N[action] += 1
        # Update action-value estimate using incremental update rule
        if self.N[action] > 0:
            self.Q[action] += (reward - self.Q[action]) / self.N[action]
        else:
            self.Q[action] = reward  # Initialize Q[action] if N[action] is 0


# Example usage:
num_arms = 10
ucb_agent = UCB(num_arms)

num_steps = 100
for t in range(num_steps):
    action = ucb_agent.select_action(t)
    reward = np.random.normal(loc=0.0, scale=1.0)  # Example reward using normal distribution
    ucb_agent.update(action, reward)
    
    # Print current state for debugging and analysis
    print(f"Time step: {t}, Action: {action}, Reward: {reward:.2f}, Q: {ucb_agent.Q}, N: {ucb_agent.N}")
