import numpy as np
import math

from calculus.third_course.support.Rotation import RotationWithBarriers
from calculus.third_course.support.checker import check_answer


class Richardson:
    def __init__(self, A: np.array, b: np.array, x: np.array, p: int = 4):
        self._tol = math.pow(10, - (p-1))
        self._eta = None
        self._tau_0 = None
        self._rho_0 = None
        self._A = A
        self._b = b
        self._lambda_min = None
        self._lambda_max = None
        self._n = 8
        self._x = x
        self._p = p

    def _compute_SZ(self):
        rotator = RotationWithBarriers(self._A, self._p)
        sz = rotator.compute()
        self._lambda_max = sz[-1]
        self._lambda_min = sz[0]

    def _compute_tau_0(self):
        self._tau_0 = 2 / (self._lambda_min + self._lambda_max)

    def _compute_eta(self):
        self._eta = self._lambda_min / self._lambda_max

    def _compute_rho_0(self):
        self._rho_0 = (1 - self._eta) / (1 + self._eta)

    def _v_k(self, k):
        return np.cos(
            (2 * k - 1) * np.pi /
            (2 * self._n)
        )

    def _t_k(self, k):
        return self._tau_0 / (1 + self._rho_0 * self._v_k(k))

    def _compute_x(self, x: np.array):
        x_k = x
        for k in range(1, self._n + 1):
            x_k = (self._b - self._A @ x_k) * self._t_k(k) + x_k

        return x_k

    def compute(self):
        self._compute_SZ()
        self._compute_eta()
        self._compute_rho_0()
        self._compute_tau_0()
        x = self._x
        iterations = 0
        while np.linalg.norm(self._A @ x - self._b) > self._tol:
            x = self._compute_x(x)
            iterations += 1
        return x, iterations * self._n


def main():
    # A = np.array([[2.2, 1, 0.5, 2],
    #               [1, 1.3, 2, 1],
    #               [0.5, 2, 0.5, 1.6],
    #               [2, 1, 1.6, 2]])
    # A = np.array([[-0.168700, 0.353699, 0.008540, 0.733624],
    #               [0.353699, 0.056519, -0.723182, -0.076440],
    #               [0.008540, -0.723182, 0.015938, 0.342333],
    #               [0.733624, -0.076440, 0.342333, -0.045744]])
    # A = np.array([[1.00, 0.42, 0.54, 0.66],
    #               [0.42, 1.00, 0.32, 0.44],
    #               [0.54, 0.32, 1.00, 0.22],
    #               [0.66, 0.44, 0.22, 1.00]])
    A = np.array([[2, 1],
                 [1, 2]])
    b = np.array([4,5])
    # b = np.array([1, 1, 1, 1])
    x = np.zeros_like(b)

    richardson = Richardson(A, b, x)

    x, iterations = richardson.compute()
    print(f"Итерационный процесс завершился за {iterations} итераций")
    check_answer(A, b, x)



if __name__ == '__main__':
    main()
