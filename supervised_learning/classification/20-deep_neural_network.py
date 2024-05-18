#!/usr/bin/env python3
""" Deep Neural Network
"""

import numpy as np


class DeepNeuralNetwork:
    """ Class that defines a deep neural network performing binary
        classification.
    """

    def __init__(self, nx, layers):
        """ Instantiation function

        Args:
            nx (int): number of input features
            layers (list): representing the number of nodes in each layer of
                           the network
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                # He et al. initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                # He et al. initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            # Zero initialization
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    # add getter method
    @property
    def L(self):
        """ Return layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """ Return dictionary with intermediate values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Return weights and bias dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation

        Args:
            X (numpy.array): Input array with
            shape (nx, m) = (featurs, no of examples)
        """
        self.cache["A0"] = X
        # print(self.cache)
        for i in range(1, self.L+1):
            # extract values
            W = self.weights['W'+str(i)]
            b = self.weights['b'+str(i)]
            A = self.cache['A'+str(i - 1)]
            # do forward propagation
            z = np.matmul(W, A) + b
            sigmoid = 1 / (1 + np.exp(-z))  # this is the output
            # store output to the cache
            self.cache["A"+str(i)] = sigmoid
        return self.cache["A"+str(i)], self.cache

    def cost(self, Y, A):
        """ Calculate the cost of the Neural Network.

        Args:
            Y (numpy.array): Actual values
            A (numpy.array): predicted values of the neural network

        Returns:
            _type_: _description_
        """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost

    def evaluate(self, X, Y):
        """ Evaluate the neural network

        Args:
            X (numpy.array): Input array
            Y (numpy.array): Actual values

        Returns:
            prediction, cost: return predictions and costs
        """
        self.forward_prop(X)
        # get output of the neural network from the cache
        output = self.cache.get("A" + str(self.L))
        return np.where(output >= 0.5, 1, 0), self.cost(Y, output)
