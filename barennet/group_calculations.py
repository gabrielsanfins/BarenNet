import numpy as np


def find_buckingham_group_exponents_matrix(A_matrix: np.ndarray,
                                           B_matrix: np.ndarray) -> np.ndarray:
    """

    Finds the Buckingham's similarity group exponents matrix Delta by solving
    the linear matrix equation B_matrix * Delta = A_matrix

    """
    delta_matrix = np.linalg.solve(B_matrix, A_matrix)
    return delta_matrix


def find_renormalization_group_exponents_matrix(
        Xi_matrix: np.ndarray,
        B_matrix: np.ndarray,
        n: int, l: int) -> np.ndarray:
    """

    Finds the Renormalization group exponents matrix. Xi is the matrix of the
    incomplete similarity exponents, B_matrix is the beta matrix and n and l
    are integers that represent the quantity of similar parameters and total
    dimensionless quantities respectively.

    COMPLETE THIS DESCRIPTION BY REFERENCING THE THESIS OR THE README FOR THE
    MATRIX CONSTRUCTIONS AND CALCULATIONS DONE HERE.

    """
    A_renorm_matrix = np.zeros(shape=(n, n))
    B_renorm_matrix = np.zeros(shape=(n, l-n))

    for i in range(n):
        beta_n_vector = B_matrix[i, :n]
        print(beta_n_vector)
        vector_sum = np.zeros(shape=(1, n))
        for j in range(l-n):
            vector_sum += Xi_matrix[i, j] * B_matrix[n+j, :n]

        print(vector_sum)

        A_renorm_matrix[i, :] = beta_n_vector + vector_sum

    print(A_renorm_matrix)

    for i in range(n):
        for j in range(l-n):
            beta = B_matrix[i, n+j]
            beta_sum = 0

            for k in range(l-n):
                beta_sum += Xi_matrix[i, k] * B_matrix[n+k, n+j]

            B_renorm_matrix[i, j] = - (beta + beta_sum)

    print(B_renorm_matrix)

    Mu_matrix = np.linalg.solve(A_renorm_matrix, B_renorm_matrix)

    return Mu_matrix
