import numpy as np


class SimpleIteration:
    def __init__(self, A: np.array, tol: float = 1e-5):
        self._tol = tol
        self._A = A

    def _compute(self, x: np.array):
        y = self._A @ x
        lmd = y @ x
        x_new = y / np.linalg.norm(y)
        return lmd, x_new

    def compute(self):
        x = np.ones(self._A.shape[0])
        eigenvalue, x_old = self._compute(x)

        while True:
            eigenvalue_new, x_new = self._compute(x_old)

            if np.linalg.norm(x_old - x_new) < self._tol:
                return eigenvalue_new, x_new

            x_old = x_new


def main():
    tol = 1e-5
    A = np.array([[1.00, 0.42, 0.54, 0.66],
                  [0.42, 1.00, 0.32, 0.44],
                  [0.54, 0.32, 1.00, 0.22],
                  [0.66, 0.44, 0.22, 1.00]])

    iterator = SimpleIteration(A, tol)
    eigenvalue, eigenvector = iterator.compute()
    print("Наибольшее собственное значение =", eigenvalue)
    print("Собственный вектор соответствующий этому значению: ", eigenvector)
    print("Проверка вида Ax - lmd * x = ", np.linalg.norm(A @ eigenvector - eigenvalue * eigenvector))


if __name__ == '__main__':
    main()
