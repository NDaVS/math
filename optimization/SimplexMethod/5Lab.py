from scipy.optimize import linprog
import numpy as np


class SimplexMethod:
    def __init__(self, matrix: np.array, b: np.array, c: np.array):
        self._matrix = matrix
        self._b = b
        self._c = c
        self._n, self._m = matrix.shape

    def _make_simplex_table(self):
        self._simplex_table = np.hstack((self._matrix, np.eye(self._n), self._b.reshape(-1, 1)))
        c_extended = np.hstack((self._c, np.zeros(self._n + 1)))
        self._simplex_table = np.vstack((self._simplex_table, -c_extended))

    def _pivot(self, row, col):
        self._simplex_table[row] /= self._simplex_table[row, col]
        for i in range(self._simplex_table.shape[0]):
            if i != row:
                self._simplex_table[i] -= self._simplex_table[i, col] * self._simplex_table[row]

    def compute(self):
        self._make_simplex_table()
        while np.any(self._simplex_table[-1, :-1] < 0):
            col = np.argmin(self._simplex_table[-1, :-1])

            if np.all(self._simplex_table[:-1, col] <= 0):
                raise ValueError("Решение неограничено.")

            ratios = self._simplex_table[:-1, -1] / self._simplex_table[:-1, col]
            ratios[self._simplex_table[:-1, col] <= 0] = np.inf
            row = np.argmin(ratios)
            self._pivot(row, col)

        solution = np.zeros(self._m)
        for i in range(self._n):
            basic_col = np.where(self._simplex_table[i, :self._m] == 1)[0]
            if len(basic_col) == 1:
                solution[basic_col[0]] = self._simplex_table[i, -1]

        objective_value = self._simplex_table[-1, -1]
        return solution, objective_value


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)

    # np.random.seed(42)
    A = np.random.randint(1, 10, size=(8, 6))
    b = np.random.randint(10, 20, size=8)
    c = np.random.randint(1, 10, size=6)

    print("Матрица ограничений (A):")
    print(A)
    print("\nПравая часть ограничений (b):")
    print(b)
    print("\nКоэффициенты целевой функции (c):")
    print(c)

    sm = SimplexMethod(A, b, c)
    try:
        solution, objective_value = sm.compute()
        print("\nПрямая задача:")
        print("Решение:", solution)
        print("Значение целевой функции:", objective_value)
    except ValueError as e:
        print("Ошибка:", e)


    # Решение двойственной задачи
    c_dual = b  # Целевая функция двойственной задачи
    A_dual = A.T  # Транспонированная матрица ограничений
    b_dual = c  # Правая часть двойственных ограничений

    res_dual = linprog(c_dual, A_ub=-A_dual, b_ub=-b_dual, bounds=(0, None), method='highs')
    print("\nДвойственная задача:")
    if res_dual.success:
        print("Оптимальное значение:", res_dual.fun)
        print("Оптимальные переменные:", res_dual.x)
    else:
        print("Ошибка:", res_dual.message)

    c_dual = b  # целевая функция двойственной задачи
    A_dual = -np.array(A).T  # транспонированные коэффициенты ограничений с отрицательным знаком
    b_dual = -np.array(c)  # коэффициенты двойственных ограничений

    res_dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(0, None), method='highs')
    print("\nРешение двойственной задачи:")
    print("Оптимальное значение:", res_dual.fun)
    print("Оптимальные переменные:", res_dual.x)
