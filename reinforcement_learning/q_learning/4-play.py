#!/usr/bin/env python3
""" Play """

import numpy as np

def play(env, Q, max_steps=100):
    """Plays an episode using the trained agent."""

    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        env.render()
        
        action = np.argmax(Q[state])
        
        # Take the action in the environment
        next_state, reward, done, _ = env.step(action)
        
        total_reward += reward 
        state = next_state
        
        if done:
            break  
    
    return total_reward
