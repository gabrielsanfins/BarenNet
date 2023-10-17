import numpy as np


def find_buckingham_group_exponents_matrix(A_matrix: np.ndarray,
                                           B_matrix: np.ndarray):
    """

    Find the Buckingham's similarity group exponents matrix Delta by solving
    the linear matrix equation B_matrix * Delta = A_matrix

    """
    delta_matrix = np.linalg.solve(B_matrix, A_matrix)
    return delta_matrix
