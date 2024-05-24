# | NAME |  ID | Department |
## Sefineh Tesfa UGR/2844/12  AI 

# Reinforcement Learning

## Overview
This project implements various reinforcement learning algorithms to solve grid world and multi-armed bandit problems. It includes implementations of algorithms such as Value Iteration, Policy Iteration, Q-Learning, Epsilon-Greedy Policy, and the Upper Confidence Bound (UCB) Algorithm.

### Algorithms Implemented
- Value Iteration
- Policy Iteration
- Q-Learning
- Epsilon-Greedy Policy
- Upper Confidence Bound (UCB) Algorithm

### Problems Addressed
1. **Grid World Problem**
   - Environment: A 2D grid with states (empty, obstacle, goal).
   - Objective: Find the shortest path to the goal while maximizing total reward.

2. **Single-State Multi-Armed Bandit Problem**
   - Environment: Each action (arm) has a probabilistic reward.
   - Objective: Maximize cumulative reward by choosing actions wisely over time.

## Files and Structure
The project includes the following files and directories:

- `grid_world/`: Contains implementations and scripts related to grid world problems.
- `multi_armed_bandit/`: Contains implementations and scripts related to multi-armed bandit problems.
- `README.md`: This file, providing an overview of the project and instructions.
- Other scripts: Various scripts implementing specific algorithms and environments.

## Setup and Dependencies
To run the project, ensure you have Python 3.x installed along with the necessary libraries:

- `numpy`: For numerical operations.
- `gym`: OpenAI Gym for reinforcement learning environments.

Install dependencies using pip:

# Usage
```bash
pip install numpy gym
```
## Running Grid World Algorithms
```
python grid_world/value_iteration.py
python grid_world/policy_iteration.py
python grid_world/q_learning.py
python grid_world/epsilon_greedy_policy.py
python grid_world/ucb.py

```
