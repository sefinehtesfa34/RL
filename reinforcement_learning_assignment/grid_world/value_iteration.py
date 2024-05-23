import numpy as np
import gym

def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    Performs value iteration to find the optimal policy and value function for the given environment.

    Args:
        env (gym.Env): The Gym environment to interact with.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        theta (float, optional): Convergence threshold. Defaults to 1e-6.

    Returns:
        numpy.ndarray: Optimal policy (best action for each state).
        numpy.ndarray: Optimal value function (expected cumulative reward from each state).
    """
    nS = env.observation_space.n  # Number of states
    nA = env.action_space.n       # Number of actions
    V = np.zeros(nS)              # Initialize value function array
    
    while True:
        delta = 0  # Initialize max change in value function across states
        for s in range(nS):
            v = V[s]  # Current value of state s
            # Update value function for state s using Bellman optimality equation
            V[s] = max([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
                        for a in range(nA)])
            # Calculate maximum change in value function across all states
            delta = max(delta, abs(v - V[s]))
        
        # If maximum change in value function is below threshold, break out of loop
        if delta < theta:
            break

    # Extract optimal policy (best action for each state)
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        # Choose action that maximizes expected cumulative reward (value function) for state s
        policy[s] = np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
                               for a in range(nA)])
    
    return policy, V

# Create FrozenLake environment with deterministic transitions (is_slippery=False)
env = gym.make('FrozenLake-v1', is_slippery=False)

# Perform value iteration to obtain optimal policy and value function
policy, V = value_iteration(env)

# Print results: Optimal Policy and Value Function reshaped into grid format
print("Optimal Policy (Value Iteration):")
print(policy.reshape(env.nrow, env.ncol))
print("Value Function:")
print(V.reshape(env.nrow, env.ncol))
