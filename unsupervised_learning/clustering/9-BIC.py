#!/usr/bin/env python3
"""This module contains a function that perfoms
finds the best number of clusters for a GMM using the
Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters for a GMM using the
    Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int):
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    if kmax is None:
        kmax = iterations

    n = X.shape[0]
    prior_bic = 0
    likelyhoods = bics = []
    best_k = kmax
    pi_prev = m_prev = S_prev = best_res = None
    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol,
                                                   verbose)
        bic = k * np.log(n) - 2 * ll
        if np.isclose(bic, prior_bic) and best_k >= k:
            best_k = k - 1
            best_res = pi_prev, m_prev, S_prev
        pi_prev, m_prev, S_prev = pi, m, S
        likelyhoods.append(ll)
        bics.append(bic)
        prior_bic = bic

    return best_k, best_res, np.asarray(likelyhoods), np.asarray(bics)
