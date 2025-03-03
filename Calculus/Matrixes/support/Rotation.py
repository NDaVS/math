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
        return eigenvalues