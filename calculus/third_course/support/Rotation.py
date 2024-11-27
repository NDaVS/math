import numpy as np


class RotationWithBarriers:
    def __init__(self, A: np.array, p: int):
        self.A = A
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
        return self.__sgn(self.A[self.i, self.j] * (self.A[self.i, self.i] - self.A[self.j, self.j])) * \
            np.sqrt(0.5 * (1 - abs(self.A[self.i, self.i] - self.A[self.j, self.j]) / self._d()))

    def _iteration(self):
        self.C = np.zeros_like(self.A)  # Явная инициализация на каждом шаге
        for k in range(self._n):
            for l in range(self._n):
                if k != self.i and k != self.j and l != self.i and l != self.j:
                    self.C[k, l] = self.A[k, l]
                elif k != self.i and k != self.j:
                    self.C[k, self.i] = self._c() * self.A[k, self.i] + self._s() * self.A[k, self.j]
                    self.C[self.i, k] = self.C[k, self.i]
                    self.C[k, self.j] = -self._s() * self.A[k, self.i] + self._c() * self.A[k, self.j]
                    self.C[self.j, k] = self.C[k, self.j]

        self.C[self.i, self.i] = (self._c() ** 2) * self.A[self.i, self.i] + \
                                 2 * self._c() * self._s() * self.A[self.i, self.j] + \
                                 (self._s() ** 2) * self.A[self.j, self.j]

        self.C[self.j, self.j] = (self._s() ** 2) * self.A[self.i, self.i] - \
                                 2 * self._c() * self._s() * self.A[self.i, self.j] + \
                                 (self._c() ** 2) * self.A[self.j, self.j]

        self.C[self.i, self.j] = 0
        self.C[self.j, self.i] = 0

    def _sigmas(self):
        self._sigma = [np.sqrt(max(self.A[i, i] for i in range(self._n))) / (10 ** p_i) for p_i in range(self.p + 1)]

    def _findIJ(self):
        max_abs_value = 0
        indexes = [0, 0]

        for i in range(self._n):
            for j in range(i + 1, self._n):
                if abs(self.A[i, j]) > max_abs_value:  # Исправление здесь
                    max_abs_value = abs(self.A[i, j])
                    indexes[0], indexes[1] = i, j

        return indexes[0], indexes[1]

    def _isEnough(self):
        return all(abs(self.A[self.i, self.j]) < sigma for sigma in self._sigma)

    def compute(self):
        self.i, self.j = self._findIJ()
        self._sigmas()

        while not self._isEnough():
            self._iteration()
            self.A = self.C.copy()
            self.i, self.j = self._findIJ()

        sz = [self.C[i, i] for i in range(self.C.shape[0])]
        sz.sort()
        return sz
