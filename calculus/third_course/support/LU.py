import numpy as np


class LU:
    def __init__(self, A: np.array,):
        self._C = None
        self._B = None
        self._A = A
    def _sum_for_b(self, B, C, i, j):
        answer = 0

        for k in range(j):
            answer += B[i][k] * C[k][j]

        return self._A[i][j] - answer

    def _sum_for_c(self, B, C, i, j):
        answer = 0

        for k in range(i):
            answer += B[i][k] * C[k][j]

        return (self._A[i][j] - answer) / B[i][i]

    def _sum_for_y(self, b, y, i):
        sum_answer = 0

        for k in range(i):
            sum_answer += self._B[i][k] * y[k]

        return (b[i] - sum_answer) / self._B[i][i]

    def _sum_for_x(self, x, y, i):
        sum_answer = 0

        for k in range(i + 1, y.shape[0]):
            sum_answer += self._C[i][k] * x[k]

        return y[i] - sum_answer

    def _find_y(self, b):
        n = self._B.shape[0]
        y = np.zeros(n)

        y[0] = b[0] / self._B[0][0]

        for i in range(1, n):
            y[i] = self._sum_for_y(b, y, i)

        return y

    def _find_x(self, y):
        x = np.zeros(self._C.shape[0])

        for i in range(self._C.shape[0] - 1, -1, -1):
            x[i] = self._sum_for_x(x, y, i)

        x[-1] = y[-1]
        return x

    def precompute(self):
        n = self._A.shape[0]
        B = np.zeros(self._A.shape)
        C = np.zeros(self._A.shape)

        for i in range(n):
            C[i][i] = 1
            B[i][0] = self._A[i][0]

            if i == 0:
                for j in range(n):
                    C[0][j] = self._A[0][j] / B[0][0]

                continue

            for j in range(n):
                if i >= j > 0:
                    B[i][j] = self._sum_for_b(B, C, i, j)

                if j > i > 0:
                    C[i][j] = self._sum_for_c(B, C, i, j)

        self._B = B
        self._C = C

    def compute(self, b: np.array):
        return self._find_x(self._find_y(b))
