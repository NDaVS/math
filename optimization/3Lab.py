import numpy as np
import sympy as sp

class NewtonMethod:
    def __init__(self, A: np.array, b: np.array):
        # Исходные данные
        self.A = A
        self.b = b
        self._n = b.shape[0]
        self._x0 = np.zeros(self._n)
        self._lmd = 1
        self._tol = 1e-3
        self._max_iter = 10000
        self._r = 10

        # SymPy переменные
        self.variables = sp.symbols(f"x:{self._n}")  # Создаем x0, x1, ..., xn-1
        self.lmd_symbol = sp.Symbol("lambda")  # Символ для Лагранжа

        # Определяем функцию f(x)
        x = sp.Matrix(self.variables)
        self.func = (0.5 * x.T @ sp.Matrix(A) @ x + sp.Matrix(b).T @ x)[0]  # Преобразуем в скаляр

        # Ограничение ||x - x0|| - r
        self.constraint = sp.sqrt(sum((xi - x0i) ** 2 for xi, x0i in zip(self.variables, self._x0))) - self._r

        # Функция Лагранжа
        self.lagrangian = self.func + self.lmd_symbol * self.constraint

        # Вычисляем символический градиент и Гессиан
        self.gradient = sp.Matrix([sp.diff(self.lagrangian, var) for var in self.variables + (self.lmd_symbol,)])
        self.hessian = sp.hessian(self.lagrangian, self.variables + (self.lmd_symbol,))

        # Компилируем численные функции для быстрого вычисления
        self.gradient_func = sp.lambdify(self.variables + (self.lmd_symbol,), self.gradient, "numpy")
        self.hessian_func = sp.lambdify(self.variables + (self.lmd_symbol,), self.hessian, "numpy")
    def f(self, x: np.array):
        return float(self.func.subs({self.variables[i]: x[i] for i in range(len(x))}))

    def lagrange_gradient(self, x: np.array, lmd: float):
        args = tuple(x) + (lmd,)
        return np.array(self.gradient_func(*args)).astype(float).flatten()

    def lagrange_hessian(self, x: np.array, lmd: float):
        args = tuple(x) + (lmd,)
        return np.array(self.hessian_func(*args)).astype(float)

    def compute(self, x: np.array):
        solutions = []

        if np.linalg.norm(np.linalg.inv(self.A) @ self.b) <= self._r:
            solutions.append(np.linalg.inv(self.A) @ self.b)

        x = x.astype(float)

        lmd = self._lmd
        for _ in range(self._max_iter):
            # Вычисляем градиент и Гессиан
            grad = self.lagrange_gradient(x, lmd)
            hess = self.lagrange_hessian(x, lmd)

            # Решаем линейную систему для обновления (x, lambda)
            delta = np.linalg.solve(hess, -grad)
            x += delta[:-1]  # Обновляем x
            lmd += delta[-1]  # Обновляем lambda

            # Проверяем сходимость
            if np.linalg.norm(delta) < self._tol:
                solutions.append(x)
                break
        print(lmd)
        return solutions

def main():
    A = np.array([
        [0, 2, 0, 0],
        [2, -6, 1, 0],
        [0, 1, -6, 0],
        [0, 0, 0, -1],
    ])
    b = np.array([1, 2, 3, 4])
    newton = NewtonMethod(A, b)
    for i in range(4):
        x = np.array([0, 0, 0, 0])
        x[i] = 10
        answer = newton.compute(x)
        print("Solutions:", answer)
        print(sum(z ** 2 for z in answer[0]))
        print("function value:", newton.f(answer[0]))

    for i in range(4):
        x = np.array([0, 0, 0, 0])
        x[i] = -10
        answer = newton.compute(x)
        print("Solutions:", answer)
        print("||X||_2:", sum(z ** 2 for z in answer[0]))
        print("function value:", newton.f(answer[0]))

if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)
    main()
