import numpy as np
import gym
from gym import wrappers

def q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Performs Q-learning to learn an optimal policy for the given environment.

    Args:
        env (gym.Env): The Gym environment to interact with.
        num_episodes (int): Number of episodes to run Q-learning.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        epsilon (float, optional): Epsilon-greedy parameter for exploration. Defaults to 0.1.

    Returns:
        numpy.ndarray: Optimal policy (best action for each state).
        numpy.ndarray: Final Q-table (state-action values).
    """
    Q = np.zeros([env.observation_space.n, env.action_space.n])  # Initialize Q-table with zeros
    
    for i in range(num_episodes):
        state, info = env.reset()  # Reset environment and get initial state
        done = False
        
        while not done:
            # Epsilon-greedy policy: choose random action with probability epsilon, otherwise exploit
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore: take a random action
            else:
                action = np.argmax(Q[state])  # Exploit: choose the best action based on Q-table
            
            # Take action and observe the next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Check if episode is done
            
            # Q-table update using Bellman equation
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            # Move to the next state
            state = next_state

        # Print progress every 1000 episodes
        if (i + 1) % 1000 == 0:
            print(f"Episode {i + 1}/{num_episodes} completed")
            
    # Extract optimal policy (best action for each state)
    policy = np.argmax(Q, axis=1)
    
    return policy, Q

# Example usage:
env = gym.make('FrozenLake-v1')
env = wrappers.TimeLimit(env, max_episode_steps=100)  # Limiting episode steps for safety

num_episodes = 10000
policy, Q = q_learning(env, num_episodes)

print("Learned policy:")
print(policy)
print("Q-table:")
print(Q)


def print_policy(policy, action_symbols, start, target):
    policy_symbols = np.array([action_symbols[action] for action in policy])
    policy_symbols[start] = 'S'  # Mark the start
    policy_symbols[target] = 'G'  # Mark the goal (target)
    policy_grid = policy_symbols.reshape((4, 4))
    for row in policy_grid:
        print(" ".join(row))

# Symbols for left, down, right, up
action_symbols = ['L', 'D', 'R', 'U']

# Start and target locations
start = 0
target = 15

print("Learned policy:")
print_policy(policy, action_symbols, start, target)
