#!/usr/bin/env python3
"""
5. Bayesian Optimization
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor
        Args:
            f: black-box function to be optimized
            X_init: np.ndarray - (t, 1) - inputs already sampled with the
                black-box function
            Y_init: np.ndarray - (t, 1) - outputs of the black-box function
                for each input in X_init
            bounds: tuple (min, max) - bounds of the space in which to look
                for the optimal point
            ac_samples: number of samples that should be analyzed during
                acquisition
            l: length parameter for the kernel
            sigma_f: standard deviation given to the output of the
                black-box function
            xsi: exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        MIN, MAX = bounds

        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(MIN, MAX, num=ac_samples)[..., np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        Uses the Expected Improvement acquisition function
        Returns: X_next, EI
        """
        mu, _ = self.gp.predict(self.gp.X)
        sample_mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            opt_mu = np.min(mu)
        else:
            opt_mu = np.max(mu)

        imp = opt_mu - sample_mu - self.xsi
        Z = imp / sigma
        EI = ((imp * norm.cdf(Z)) + (sigma * norm.pdf(Z)))
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, np.array(EI)

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        Args:
            iterations: maximum number of iterations to perform

        Returns: X_opt, Y_opt
        """
        for i in range(iterations):
            X_next, _ = self.acquisition()

            if X_next in self.gp.X:
                break

            Y = self.f(X_next)
            self.gp.update(X_next, Y)

        idx = np.argmin(self.gp.Y)
        X_opt = self.gp.X[idx]
        Y_opt = np.array(self.gp.Y[idx])
        return X_opt, Y_opt
