import numpy as np

from calculus.third_course.support.LU import LU


class ReverseIterations:
    def __init__(self, A: np.array, tol: float = 1e-5):
        self._A = A
        self._alpha = None
        self._tol = tol

    def compute(self, x0: np.array):
        lu = LU(self._A)
        lu.precompute()
        self._alpha = np.linalg.norm(x0, ord=np.inf)
        x_old = x0
        while True:
            x_new = lu.compute(x_old / self._alpha)
            alpha_new = np.linalg.norm(x_new, ord=np.inf)

            if np.abs(alpha_new - self._alpha) < self._tol:
                return 1 / alpha_new, x_new

            self._alpha = alpha_new
            x_old = x_new


def result(matrix, tol: float):
    for A in matrix:
        print("A= \n", A)
        iterator = ReverseIterations(A, tol)
        x0 = np.ones(A.shape[0])
        eigenvalue, eigenvector = iterator.compute(x0)
        print("Наибольшее собственное значение =", eigenvalue)
        print("Собственный вектор соответствующий этому значению: ", eigenvector)
        print("Проверка вида Ax - lmd * x = ", np.linalg.norm(A @ eigenvector - eigenvalue * eigenvector))
        print("=" * 50)

def main():
    tol = 1e-5
    A1 = np.array([[-0.168700, 0.353699, 0.008540, 0.733624],
                  [0.353699, 0.056519, -0.723182, -0.076440],
                  [0.008540, -0.723182, 0.015938, 0.342333],
                  [0.733624, -0.076440, 0.342333, -0.045744]])


    A2 = np.array([[1.00, 0.42, 0.54, 0.66],
                  [0.42, 1.00, 0.32, 0.44],
                  [0.54, 0.32, 1.00, 0.22],
                  [0.66, 0.44, 0.22, 1.00]])

    A3 = np.array([[2.2, 1, 0.5, 2],
                  [1, 1.3, 2, 1],
                  [0.5, 2, 0.5, 1.6],
                  [2, 1, 1.6, 2]])
    result([A1, A2, A3], tol)


if __name__ == '__main__':
    main()
