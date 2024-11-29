import math

import numpy as np


class RotationWithBarriers:
    def __init__(self, A: np.array, p: int):
        self.A = A.copy()
        self.C = np.zeros_like(A)
        self._n = A.shape[0]
        self.p = p

        self.i = None
        self.j = None
        self._sigma = None

    def _isPositiveDefine(self):
        if self.A.shape[0] != self.A.shape[1]:
            return False

        if not np.allclose(self.A, self.A.T):
            return False

        minors = [np.linalg.det(self.A[:i, :i]) for i in range(1, self.A.shape[0] + 1)]
        return all(minor > 0 for minor in minors)

    def __sgn(self, value):
        return 1 if value >= 0 else -1

    def _d(self):
        return np.sqrt((self.A[self.i, self.i] - self.A[self.j, self.j]) ** 2 + 4 * self.A[self.i, self.j] ** 2)

    def _c(self):
        return np.sqrt(0.5 * (1 + abs(self.A[self.i, self.i] - self.A[self.j, self.j]) / self._d()))

    def _s(self):
        return (
            self.__sgn(self.A[self.i, self.j] * (self.A[self.i, self.i] - self.A[self.j, self.j]))
            * np.sqrt(0.5 * (1 - abs(self.A[self.i, self.i] - self.A[self.j, self.j]) / self._d()))
        )

    def _iteration(self):
        c = self._c()
        s = self._s()
        C = self.A.copy()

        for k in range(self._n):
            if k != self.i and k != self.j:
                C[k, self.i] = c * self.A[k, self.i] + s * self.A[k, self.j]
                C[self.i, k] = C[k, self.i]
                C[k, self.j] = -s * self.A[k, self.i] + c * self.A[k, self.j]
                C[self.j, k] = C[k, self.j]

        C[self.i, self.i] = c**2 * self.A[self.i, self.i] + 2 * c * s * self.A[self.i, self.j] + s**2 * self.A[self.j, self.j]
        C[self.j, self.j] = s**2 * self.A[self.i, self.i] - 2 * c * s * self.A[self.i, self.j] + c**2 * self.A[self.j, self.j]
        C[self.i, self.j] = 0
        C[self.j, self.i] = 0

        self.A = C.copy()

    def _sigmas(self):
        """Compute the stopping threshold values."""
        self._sigma = [np.sqrt(max(abs(self.A[i, i]) for i in range(self._n))) / (10**p_i) for p_i in range(self.p + 1)]

    def _findIJ(self):
        max_abs_value = 0
        indexes = (0, 0)

        for i in range(self._n):
            for j in range(i + 1, self._n):
                if abs(self.A[i, j]) > max_abs_value:
                    max_abs_value = abs(self.A[i, j])
                    indexes = (i, j)

        return indexes

    def _isEnough(self):
        for i in range(self._n):
            for j in range(i + 1, self._n):
                if abs(self.A[i, j]) > min(self._sigma):
                    return False
        return True

    def compute(self):
        self._sigmas()
        iterations = 0

        while not self._isEnough():
            self.i, self.j = self._findIJ()
            self._iteration()
            iterations += 1

        eigenvalues = [self.A[i, i] for i in range(self._n)]
        eigenvalues.sort()
        return eigenvalues, iterations


def main():
    isPassed = True
    # A = np.array([[2.2, 1, 0.5, 2],
    #               [1, 1.3, 2, 1],
    #               [0.5, 2, 0.5, 1.6],
    #               [2, 1, 1.6, 2]])
    # A = np.array([[-0.168700, 0.353699, 0.008540, 0.733624],
    #               [0.353699, 0.056519, -0.723182, -0.076440],
    #               [0.008540, -0.723182, 0.015938, 0.342333],
    #               [0.733624, -0.076440, 0.342333, -0.045744]])
    A = np.array([[1.00, 0.42, 0.54, 0.66],
                  [0.42, 1.00, 0.32, 0.44],
                  [0.54, 0.32, 1.00, 0.22],
                  [0.66, 0.44, 0.22, 1.00]])
    A = np.array([[2, 1],
                 [1, 2]])
    p = 2
    rotation = RotationWithBarriers(A, p)
    solution, iterations = rotation.compute()

    eigenvalues = solution
    print(f"Converged in {iterations} iterations.")
    print(f"Eigenvalues: {eigenvalues}")

    for i in range(A.shape[0]):
        if np.linalg.det(A - eigenvalues[i] * np.eye(A.shape[0])) < math.pow(10, - p):
            print("Success")

        else:
            print(f"Fail! Error = {np.linalg.det(A - eigenvalues[i] * np.eye(A.shape[0]))}")
            isPassed = False
            continue

    if isPassed:
        print(f"All values has been tested successfully.")
        return

    print("One or more value is wrong or error is too high")


if __name__ == "__main__":
    np.set_printoptions(linewidth=200, suppress=True)
    main()
