#!/usr/bin/env python3
""" Q-Learning """

import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning."""
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            # Update reward if the agent falls into a hole
            if done and reward == 0:
                reward = -1
            
            # Update Q-table using the Q-learning formula
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            total_reward += reward  # Accumulate reward
            state = next_state  # Move to the next state
            
            if done:
                break  
        
        total_rewards.append(total_reward)  # Store total reward for the episode
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    
    return Q, total_rewards
