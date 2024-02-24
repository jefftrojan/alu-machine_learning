#!/usr/bin/env python3
"""Function that calculates the inverse of a matrix"""


def determinant(matrix):
    """Function that calculates the determinant of a matrix"""
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
    det = []
    for i in range(len(matrix)):
        mini = [[j for j in matrix[i]] for i in range(1, len(matrix))]
        for j in range(len(mini)):
            mini[j].pop(i)
        if i % 2 == 0:
            det.append(matrix[0][i] * determinant(mini))
        if i % 2 == 1:
            det.append(-1 * matrix[0][i] * determinant(mini))
    return sum(det)


def cofactor(matrix):
    """Function that calculates the cofactor matrix of a matrix"""
    if len(matrix) == 1 and len(matrix[i]) == 1:
        return [[1]]
    if len(matrix) == 2:
        cofactor = [i[::-1] for i in matrix]
        cofactor = cofactor[::-1]
        cofactor = [[cofactor[i][j] if (i + j) % 2 == 0 else -cofactor[i][j]
                     for j in range(len(cofactor[i]))]
                    for i in range(len(cofactor))]
        return cofactor
    cofactor = [[j for j in matrix[i]] for i in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            mini = [[j for j in matrix[i]] for i in range(len(matrix))]
            mini = mini[:i] + mini[i + 1:]
            for k in range(len(mini)):
                mini[k].pop(j)
            if (i + j) % 2 == 0:
                cofactor[i][j] = determinant(mini)
            if (i + j) % 2 == 1:
                cofactor[i][j] = -1 * determinant(mini)
    return cofactor


def inverse(matrix):
    """Function that calculates the inverse of a matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")
    for i in matrix:
        if len(matrix) != len(i):
            raise ValueError("matrix must be a non-empty square matrix")
    det = determinant(matrix)
    if det == 0:
        return None
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1 / det]]
    adjugate = cofactor(matrix)
    copy = [[j for j in adjugate[i]] for i in range(len(adjugate))]
    for i in range(len(adjugate)):
        for j in range(len(adjugate[i])):
            adjugate[j][i] = copy[i][j]
    inverse = [[j / det for j in i] for i in adjugate]
    return inverse
