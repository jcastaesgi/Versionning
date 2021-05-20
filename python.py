## Fonction gaussian elimination - python ##

"""
Gaussian elimination method for solving a system of linear equations.
Gaussian elimination - https://en.wikipedia.org/wiki/Gaussian_elimination
"""


import numpy as np


def retroactive_resolution(coefficients: np.matrix, vector: np.ndarray) -> np.ndarray:
    """
    This function performs a retroactive linear system resolution
        for triangular matrix
    Examples:
        2x1 + 2x2 - 1x3 = 5         2x1 + 2x2 = -1
        0x1 - 2x2 - 1x3 = -7        0x1 - 2x2 = -1
        0x1 + 0x2 + 5x3 = 15

>>> gaussian_elimination([[2, 2, -1], [0, -2, -1], [0, 0, 5]], [[5], [-7], [15]])
    array([[2.],
           [2.],
           [3.]])
    >>> gaussian_elimination([[2, 2], [0, -2]], [[-1], [-1]])
    array([[-1. ],
           [ 0.5]])
    """

    rows, columns = np.shape(coefficients)

    x = np.zeros((rows, 1), dtype=float)
    for row in reversed(range(rows)):
        sum = 0
        for col in range(row + 1, columns):
            sum += coefficients[row, col] * x[col]

        x[row, 0] = (vector[row] - sum) / coefficients[row, row]

    return x
