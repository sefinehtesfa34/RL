import numpy as np

class EpsilonGreedyBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit algorithm implementation.

    Attributes:
        k (int): Number of arms (actions).
        epsilon (float): Probability of exploration (choosing a random action).
        counts (numpy.ndarray): Array to track the number of times each action has been chosen.
        values (numpy.ndarray): Array to track the estimated value (reward) of each action.
    """

    def __init__(self, k, epsilon):
        """
        Initializes the EpsilonGreedyBandit with given number of arms and epsilon value.

        Args:
            k (int): Number of arms (actions).
            epsilon (float): Probability of exploration (choosing a random action).
        """
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k)    # Initialize counts for each action to zero
        self.values = np.zeros(k)    # Initialize estimated values for each action to zero

    def select_action(self):
        """
        Selects an action based on the epsilon-greedy strategy.

        Returns:
            int: Chosen action index.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)   # Randomly choose an action
        else:
            return np.argmax(self.values)      # Choose the action with the highest estimated value

    def update(self, action, reward):
        """
        Updates the action-value estimates based on the received reward.

        Args:
            action (int): Index of the action chosen.
            reward (float): Reward received from the environment.
        """
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

def run_bandit(bandit, env, num_episodes):
    """
    Runs the multi-armed bandit algorithm for a specified number of episodes.

    Args:
        bandit (EpsilonGreedyBandit): The bandit instance to run.
        env: Bandit environment that supports 'step' method for actions.
        num_episodes (int): Number of episodes to run.

    Returns:
        float: Total accumulated reward across all episodes.
    """
    total_reward = 0
    for episode in range(num_episodes):
        action = bandit.select_action()       # Select action using bandit's strategy
        reward = env.step(action)             # Get reward from environment for chosen action
        bandit.update(action, reward)         # Update bandit's action-value estimates
        total_reward += reward                # Accumulate total reward
    return total_reward

# Example environment for multi-armed bandit
class BanditEnv:
    """
    Example Bandit environment where each action has a probabilistic reward.

    Attributes:
        k (int): Number of arms (actions).
        probs (numpy.ndarray): Array of probabilities for each action.
    """

    def __init__(self, k):
        """
        Initializes the BanditEnv with a given number of arms.

        Args:
            k (int): Number of arms (actions).
        """
        self.k = k
        self.probs = np.random.rand(k)   # Initialize probabilities for each action randomly

    def step(self, action):
        """
        Executes the chosen action and returns a reward based on its probability.

        Args:
            action (int): Index of the action to take.

        Returns:
            int: 1 if action results in reward, otherwise 0.
        """
        return 1 if np.random.rand() < self.probs[action] else 0   # Return reward based on action's probability

# Example usage
k = 10
env = BanditEnv(k)                      # Initialize Bandit environment with 10 arms
bandit = EpsilonGreedyBandit(k, epsilon=0.1)  # Initialize Epsilon-Greedy Bandit with epsilon=0.1
total_reward = run_bandit(bandit, env, 1000)  # Run bandit algorithm for 1000 episodes
print("Total Reward (Epsilon-Greedy):", total_reward)  # Print total accumulated reward
