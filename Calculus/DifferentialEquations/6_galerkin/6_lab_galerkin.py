import numpy as np
import sympy as sp
from prettytable import PrettyTable


class Galerkin:
    def __init__(self, x, c1, c2, a, b, n):
        self.x = x
        self.a = a
        self.b = b
        self.n = n
        self.c1 = c1
        self.c2 = c2

    def _phi(self, k):
        """Базисные функции."""
        if k == 0:
            return 1-0.5 * self.x
        return (1-self.x ) * self.x** (k)
    def _P(self):
        return -(1 + self.x)
    def _Q(self):
        """Коэффициент Q(x) в уравнении."""
        return -1

    def _F(self):
        """Правая часть F(x) в уравнении."""
        return 2/ ((1 + self.x ) ** 3)



    def u(self):
        """Приближённое решение u(x)."""
        return self._phi(0) + self.c1 * self._phi(1) + self.c2 * self._phi(2)

    def residual(self):
        """Невязка уравнения."""
        u = self.u()

        return sp.diff(u, self.x, 2) + self._P() * sp.diff(u, self.x) + self._Q() * u - self._F()

    def dPhi_dCi(self, i):
        """Уравнение Галёркина для коэффициента c_i."""
        phi_i = self._phi(i)
        return sp.integrate(self.residual() * phi_i, (self.x, self.a, self.b))


def main():
    n = 3
    a, b = 0, 1
    x = sp.symbols('x')
    c1, c2 = sp.symbols('c1 c2')
    galerkin = Galerkin(x, c1, c2, a, b, n)

    # Формируем систему уравнений
    A = np.zeros((n - 1, n - 1))
    b = np.zeros(n - 1)
    for i in range(1, n):
        expr = galerkin.dPhi_dCi(i)
        print(expr)
        coeff_c1 = expr.coeff(c1)
        coeff_c2 = expr.coeff(c2)
        const_term = expr.subs({c1: 0, c2: 0})
        A[i - 1][0] = coeff_c1
        A[i - 1][1] = coeff_c2
        b[i - 1] = -const_term

    solution = np.linalg.solve(A, b)
    answer = galerkin.u().subs(c1, solution[0]).subs(c2, solution[1])
    #
    # # Вывод результатов
    ab = np.linspace(0, 1, 10)
    my_values = [answer.subs(x, xi) for xi in ab]
    true_values = [1 / (xi + 1) for xi in ab]
    errors = [abs(true_values[i] - my_values[i]) for i in range(10)]

    table = PrettyTable()
    table.field_names = ["x", "Истинное значение", "Галёркин", "Ошибка"]

    for i in range(len(ab)):
        table.add_row([f"{ab[i]:.8f}", f"{true_values[i]:.12f}", f"{my_values[i]:.12f}", f"{errors[i]:.12f}"])

    print(table)


if __name__ == '__main__':
    main()