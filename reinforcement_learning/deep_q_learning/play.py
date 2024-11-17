#!/usr/bin/env python3
""" Play """

import gym
import numpy as np
from keras.models import load_model
from rl.agents import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

env = gym.make('Breakout-v0')

model = load_model('policy.h5')

memory = SequentialMemory(limit=50000, window_length=1)
policy = GreedyQPolicy()

dqn = DQNAgent(model=model, memory=memory, policy=policy, 
               nb_actions=env.action_space.n)

# Play the game
state = env.reset()
done = False
total_reward = 0

while not done:
    env.render()  
    action = dqn.forward(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward 

print(f'Total Reward: {total_reward}')
env.close()
