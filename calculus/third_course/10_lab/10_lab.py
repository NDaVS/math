from calculus.third_course.support.checker import *


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

        # Проверяем, что матрица квадратная
        if self.A.shape[0] != self.A.shape[1]:
            return False

        # Проверяем, что матрица симметрична
        if not np.allclose(self.A, self.A.T):
            return False

        # Вычисляем главные миноры
        minors = [np.linalg.det(self.A[:i, :i]) for i in range(1, self.A.shape[0] + 1)]

        # Проверяем, что все главные миноры положительны
        return all(minor > 0 for minor in minors)

    def __sgn(self, value):
        if value >= 0:
            return 1

        return -1

    def _d(self, ):
        return np.sqrt((self.A[self.i, self.i] - self.A[self.j, self.j]) ** 2 + 4 * self.A[self.i, self.j] ** 2)

    def _c(self):
        return np.sqrt(0.5 * (1 + np.abs(self.A[self.i, self.i] - self.A[self.j, self.j]) / self._d()))

    def _s(self):
        return self.__sgn(self.A[self.i, self.j] * (self.A[self.i, self.i] - self.A[self.j, self.j])) * \
            np.sqrt(
                0.5 * (1 - np.abs(self.A[self.i, self.i] - self.A[self.j, self.j]) / self._d())
            )

    def _iteration(self):
        for k in range(self._n):
            for l in range(self._n):
                if k != self.i and k != self.j and l != self.i and l != self.j:
                    self.C[k, l] = self.A[k, l]

                elif k != self.i and k != self.j:
                    self.C[k, self.i] = self._c() * self.A[k, self.i] + self._s() * self.A[k, self.j]
                    self.C[self.i, k] = self._c() * self.A[k, self.i] + self._s() * self.A[k, self.j]

                    self.C[k, self.j] = (-1) * self._s() * self.A[k, self.i] + self._c() * self.A[k, self.j]
                    self.C[self.j, k] = (-1) * self._s() * self.A[k, self.i] + self._c() * self.A[k, self.j]

        self.C[self.i, self.i] = (self._c() ** 2) * self.A[self.i, self.i] + \
                                 2 * self._c() * self._s() * self.A[self.i, self.j] + \
                                 (self._s() ** 2) * self.A[self.j, self.j]

        self.C[self.j, self.j] = (self._s() ** 2) * self.A[self.i, self.i] - \
                                 2 * self._c() * self._s() * self.A[self.i, self.j] + \
                                 (self._c() ** 2) * self.A[self.j, self.j]

        self.C[self.i, self.j] = 0
        self.C[self.j, self.i] = 0

    def _sigmas(self):
        self._sigma = [np.sqrt(max(abs(self.A[i, i]) for i in range(self._n))) * p_i for p_i in range(1, self.p)]

    def _findIJ(self):
        max_abs_value = 0
        indexes = [0, 0]

        for i in range(1, self._n):
            for j in range(i + 1, self._n):
                if self.A[i, j] > max_abs_value:
                    max_abs_value = self.A[i, j]
                    indexes[0] = i
                    indexes[1] = j

        return indexes[0], indexes[1]

    def _isEnough(self):
        if all(self.A[self.i, self.j] < sigma for sigma in self._sigma):
            return True

        return False

    def compute(self):
        self.i, self.j = self._findIJ()
        self._sigmas()

        while not self._isEnough():
            self.i, self.j = self._findIJ()
            self._iteration()


def main():
    A = np.array([
        [10.9, 1.2, 2.1, 0.9],
        [1.2, 11.2, 1.5, 2.5],
        [2.1, 1.5, 9.8, 1.3],
        [0.9, 2.5, 1.3, 12.1],
    ])

    A1 = np.array([
        [3.82, 1.02, 0.75, 0.81],
        [1.05, 4.53, 0.98, 1.53],
        [0.73, 0.85, 4.71, 0.81],
        [0.88, 0.81, 1.28, 3.50]
    ], dtype=float)

    solution = RotationWithBarriers(A, 5)


if __name__ == '__main__':
    np.set_printoptions(linewidth=200,
                        suppress=True)  # Установите ширину строки и отключите экспоненциальный формат

    main()
