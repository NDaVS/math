import numpy as np
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, A: np.array, b: np.array):
        self._A = A.astype(float)
        self._n = A.shape[0]
        self._A_resolve = None
        self._b = b.astype(float)
        self._perm = list(range(self._n))

    def solveGaussWithAllMax(self):
        self._A_resolve = self._A.copy()  # Create a copy of the matrix to avoid modifying the original
        operations = 0
        for k in range(self._n - 1):
            # Find the maximum element
            max_row, max_col = divmod(np.abs(self._A_resolve[k:, k:]).argmax(), self._n - k)
            operations += (self._n - k) ** 2
            max_row += k
            max_col += k

            # Swap rows
            if max_row != k:
                self._A_resolve[[k, max_row]] = self._A_resolve[[max_row, k]]
                self._b[[k, max_row]] = self._b[[max_row, k]]

            # Swap columns
            if max_col != k:
                self._A_resolve[:, [k, max_col]] = self._A_resolve[:, [max_col, k]]
                self._perm[k], self._perm[max_col] = self._perm[max_col], self._perm[k]
            operations += 1

            # Row transformations
            for i in range(k + 1, self._n):
                factor = self._A_resolve[i, k] / self._A_resolve[k, k]
                self._A_resolve[i, k:] -= factor * self._A_resolve[k, k:]
                self._b[i] -= factor * self._b[k]
                operations += 1

        # Back substitution
        x = np.zeros(self._n)
        for i in range(self._n - 1, -1, -1):
            x[i] = (self._b[i] - np.dot(self._A_resolve[i, i + 1:], x[i + 1:])) / self._A_resolve[i, i]
            operations += 1

        # Adjust solution based on column permutations
        x_final = np.zeros(self._n)
        for i, p in enumerate(self._perm):
            x_final[p] = x[i]
            operations += 1

        return x_final, operations


def main():
    sizes = list(range(5, 105, 5))  # Matrix sizes
    operations_list = []  # To store the number of operations
    errors = []  # To store errors

    for n in sizes:
        A = np.random.randint(1, 10, size=(n, n))
        b = np.random.randint(10, 20, size=n)
        solver = Solver(A, b)

        solution, operations = solver.solveGaussWithAllMax()
        operations_list.append(operations)

        # Calculate error
        error = np.linalg.norm(A @ solution - b)
        errors.append(error)

    # Plotting graphs
    plt.figure(figsize=(12, 6))

    # Graph of the number of operations
    plt.subplot(1, 2, 1)
    plt.plot(sizes, operations_list, marker='o', label='Operations')
    plt.title('Number of Operations vs Matrix Size')
    plt.xlabel('Matrix Size')
    plt.ylabel('Number of Operations')
    plt.grid()
    plt.legend()

    # Graph of errors
    plt.subplot(1, 2, 2)
    plt.plot(sizes, errors, marker='s', label='Error', color='red')
    plt.title('Error vs Matrix Size')
    plt.xlabel('Matrix Size')
    plt.ylabel('Error')
    plt.grid()
    plt.legend()

    # Show plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
