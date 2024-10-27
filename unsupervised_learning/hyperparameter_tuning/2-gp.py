#!/usr/bin/env python3

"""
This script represents a noiseless 1D Gaussian process
"""

import numpy as np


class GaussianProcess:
    """
    A class that represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        A function that initializes a noiseless 1D Gaussian process
        """

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices
        """

        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(
            X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        predicts the mean and standard deviation of
        points in a Gaussian process

        X_s: numpy.ndarray of shape (s, 1) containing all of the points
            - whose mean and standard deviation should be calculated
            - s is the number of sample points
        Return mu, sigma
        - mu:
        numpy.ndarray (s,) containing the mean for each point in X_s
        """

        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        K_ss = self.kernel(X_s, X_s)
        sigma = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu, np.diag(sigma)

    def update(self, X_new, Y_new):
        """
        A function that updates a Gaussian process
        """

        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
