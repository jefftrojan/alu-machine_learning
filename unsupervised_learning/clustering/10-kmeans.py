#!/usr/bin/env python3
"""This module contains a function that perfoms
performs K-means on a dataset"""
import sklearn.cluster


def kmeans(X, k):
    """
    performs K-means on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    Returns: C, clss
        C: is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster
        clss: is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """
    k_model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    clss = k_model.labels_
    C = k_model.cluster_centers_

    return C, clss
