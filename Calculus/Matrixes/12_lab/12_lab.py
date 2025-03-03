import numpy as np


class SimpleIteration:
    def __init__(self, A: np.array, tol: float = 1e-5, max_iter: int = 1000):
        self._tol = tol
        self._A = A
        self._max_iter = max_iter

    def _compute(self, x: np.array):
        y = self._A @ x
        eigenvalue = y @ x
        x_new = y / np.linalg.norm(y)
        return eigenvalue, x_new

    def compute(self):
        x = np.ones(self._A.shape[0])
        x = x / np.linalg.norm(x)
        eigenvalue_old = 0
        iter_count = 0

        while True:
            iter_count += 1
            eigenvalue, x_new = self._compute(x)

            if (
                    np.linalg.norm(np.sign(eigenvalue) * x - x_new) < self._tol
                    and abs(eigenvalue - eigenvalue_old) < self._tol
            ):
                print(f"Метод завершился за {iter_count} итераций.")
                return eigenvalue, x_new

            x = x_new
            eigenvalue_old = eigenvalue


def result(matrix, tol: float):
    for A in matrix:
        print("A= \n", A)
        iterator = SimpleIteration(A, tol)
        eigenvalue, eigenvector = iterator.compute()
        print("Наибольшее по модулю собственное значение =", eigenvalue)
        print("Собственный вектор соответствующий этому значению: ", eigenvector)
        print("Проверка вида Ax - lmd * x = ", np.linalg.norm(A @ eigenvector - eigenvalue * eigenvector))
        print("=" * 50)


def main():
    tol = 1e-8
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
