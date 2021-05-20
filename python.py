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


