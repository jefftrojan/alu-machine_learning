#!/usr/bin/env python3

"""
This script represents a Bayesian optimization on a
noiseless 1D Gaussian process
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    A class that represents a Bayesian optimization on a
    noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes Bayesian Optimization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)
        with np.errstate(divide='warn'):
            if self.minimize:
                mu_sample_opt = np.min(self.gp.Y)
                imp = mu_sample_opt - mu - self.xsi
            else:
                mu_sample_opt = np.max(self.gp.Y)
                imp = mu - mu_sample_opt - self.xsi
            Z = imp / sigma
            ei = imp * self._cdf(Z) + sigma * self._pdf(Z)
            ei[sigma == 0.0] = 0.0
        return self.X_s[np.argmax(ei)], ei

    def _cdf(self, Z):
        """
        Computes cumulative distribution function (CDF)
        for standard normal distribution
        """
        return 0.5 * (1 + np.erf(Z / np.sqrt(2)))

    def _pdf(self, Z):
        """
        Computes the probability density function (PDF)
        for standard normal distribution
        """
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Z**2)

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        """
        X_all = []
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            X_all.append(X_next)
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        return self.gp.X[idx], self.gp.Y[idx]
