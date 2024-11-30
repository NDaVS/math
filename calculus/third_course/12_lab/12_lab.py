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
                    np.linalg.norm(np.sign(eigenvalue)*x - x_new) < self._tol
                    and abs(eigenvalue - eigenvalue_old) < self._tol
            ):
                print(f"Метод завершился за {iter_count} итераций.")
                return eigenvalue, x_new

            x = x_new
            eigenvalue_old = eigenvalue


def main():
    tol = 1e-5
    A = np.array([[-0.168700, 0.353699, 0.008540, 0.733624],
                  [0.353699, 0.056519, -0.723182, -0.076440],
                  [0.008540, -0.723182, 0.015938, 0.342333],
                  [0.733624, -0.076440, 0.342333, -0.045744]])

    iterator = SimpleIteration(A, tol)
    eigenvalue, eigenvector = iterator.compute()
    print("Наибольшее собственное значение =", eigenvalue)
    print("Собственный вектор соответствующий этому значению: ", eigenvector)
    print("Проверка вида Ax - lmd * x = ", np.linalg.norm(A @ eigenvector - eigenvalue * eigenvector))


if __name__ == '__main__':
    main()
