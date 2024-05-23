import numpy as np
import gym

def policy_evaluation(policy, env, gamma=0.99, theta=1e-6):
    """
    Evaluates a given policy by iteratively calculating the state-value function.

    Args:
        policy (numpy.ndarray): Policy to be evaluated, mapping states to actions.
        env (gym.Env): The Gym environment to interact with.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
        theta (float, optional): Convergence threshold. Defaults to 1e-6.

    Returns:
        numpy.ndarray: State-value function V(s) under the given policy.
    """
    nS = env.observation_space.n  # Number of states
    nA = env.action_space.n       # Number of actions
    V = np.zeros(nS)              # Initialize value function array
    
    while True:
        delta = 0  # Initialize max change in value function across states
        for s in range(nS):
            v = V[s]  # Current value of state s
            # Update value function V(s) using Bellman expectation equation for a given policy
            V[s] = sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][policy[s]]])
            # Calculate maximum change in value function across all states
            delta = max(delta, abs(v - V[s]))
        # If maximum change in value function is below threshold, break out of loop
        if delta < theta:
            break
    return V

def policy_iteration(env, gamma=0.99):
    """
    Performs policy iteration to find the optimal policy and value function for the given environment.

    Args:
        env (gym.Env): The Gym environment to interact with.
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.

    Returns:
        numpy.ndarray: Optimal policy (best action for each state).
        numpy.ndarray: Optimal value function (expected cumulative reward from each state).
    """
    nS = env.observation_space.n     # Number of states
    nA = env.action_space.n          # Number of actions
    policy = np.random.choice(nA, size=(nS))  # Initialize a random policy
    
    while True:
        # Evaluate the current policy
        V = policy_evaluation(policy, env, gamma)
        
        policy_stable = True  # Flag to track if policy improvement has stopped
        for s in range(nS):
            old_action = policy[s]  # Current action under the policy
            # Improve policy by selecting action that maximizes expected cumulative reward (V) for state s
            policy[s] = np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
                                   for a in range(nA)])
            # If policy changes for any state, indicate instability for further iteration
            if old_action != policy[s]:
                policy_stable = False
        
        # If policy is stable (no changes in iteration), break out of loop
        if policy_stable:
            break
    
    return policy, V

# Create FrozenLake environment with deterministic transitions (is_slippery=False)
env = gym.make('FrozenLake-v1', is_slippery=False)

# Perform policy iteration to obtain optimal policy and value function
policy, V = policy_iteration(env)

# Print results: Optimal Policy and Value Function reshaped into grid format
print("Optimal Policy (Policy Iteration):")
print(policy.reshape(env.nrow, env.ncol))
print("Value Function:")
print(V.reshape(env.nrow, env.ncol))
