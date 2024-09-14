#!/usr/bin/env python3
'''
LSTM Cell
'''


import numpy as np


class LSTMCell:
    '''
    Class that represents a cell of a LSTM
    '''
    def __init__(self, i, h, o):
        '''
        Class constructor
        '''
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        ''' Method that performs forward propagation for one time step '''
        h_x = np.hstack((h_prev, x_t))
        f = 1 / (1 + np.exp(-(np.dot(h_x, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.dot(h_x, self.Wu) + self.bu)))
        o = 1 / (1 + np.exp(-(np.dot(h_x, self.Wo) + self.bo)))
        c_tilde = np.tanh(np.dot(h_x, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_tilde
        h_next = o * np.tanh(c_next)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, c_next, y
