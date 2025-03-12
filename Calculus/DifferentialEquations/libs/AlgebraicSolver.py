import numpy as np
from numpy import ndarray


import numpy as np
from numpy import ndarray


class AlgebraicSolver:
    """
    A class to solve algebraic equations using the Thomas algorithm (TDMA).

    Methods:
    --------
    tdma(matrix: ndarray, f: ndarray) -> ndarray
        Solves a tridiagonal system of linear equations.
    """

    def tdma(self, matrix: ndarray, f: ndarray) -> ndarray:
        """
        Solves a tridiagonal system of linear equations using the Thomas algorithm.

        Parameters:
        -----------
        matrix : ndarray
            Coefficient matrix of the system.
        f : ndarray
            Right-hand side vector.

        Returns:
        --------
        ndarray
            Solution vector.
        """
        n = len(f)
        a = [0] + [matrix[i][i - 1] for i in range(1, n)]
        b = [matrix[i][i] for i in range(n)]
        c = [matrix[i][i + 1] for i in range(n - 1)] + [0]
        f = list(map(float, f))

        alpha = [-c[0] / b[0]]
        beta = [f[0] / b[0]]
        x: ndarray = np.zeros(n)
        for i in range(1, n):
            alpha.append(-c[i] / (a[i] * alpha[i - 1] + b[i]))
            beta.append((f[i] - a[i] * beta[i - 1]) / (a[i] * alpha[i - 1] + b[i]))

        x[n - 1] = beta[n - 1]

        for i in range(n - 1, 0, -1):
            x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

        return x
