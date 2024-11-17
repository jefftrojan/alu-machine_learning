#!/usr/bin/env python3
""" Train """

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

env = gym.make('Breakout-v0')

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))


model.compile(optimizer=Adam(lr=0.001), loss='mse')


memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(eps=1.0)


dqn = DQNAgent(model=model, memory=memory, policy=policy, 
               nb_actions=env.action_space.n, nb_steps_warmup=1000,
               target_model_update=1e-2)


dqn.compile(Adam(lr=0.001), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)


dqn.save_weights('policy.h5', overwrite=True)

