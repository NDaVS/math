import numpy as np
import sympy as sp


class Ritz:
    def __init__(self, x, c1, c2, a, b, n):
        self.x = x
        self.a = a
        self.b = b
        self.n = n
        self.c1 = c1
        self.c2 = c2

    def _phi(self, k):
        return 1 - 0.5 * self.x ** (2 *k)

    def _P(self):
        xi = sp.symbols('xi')
        return sp.integrate(4 * xi / (1 + xi ** 2), (xi, 0, self.x))

    def _Q(self):
        return -1 / (1 + self.x ** 2)

    def _F(self):
        return -3 / ((1 + self.x ** 2) ** 2)

    def _p(self):
        return sp.exp(self._P())

    def _q(self):
        return self._p() * self._Q()

    def _f(self):
        return self._p() * self._F()

    def _J_1(self, u):
        return self._p() * (u ** 2) - self._q() * (u ** 2) + 2 * self._f() * u

    def u(self):
        return self.c1 * self._phi(1) + self.c2 * self._phi(2)

    def dPhi_dCi(self, i):
        phi1 = self._phi(1)
        phi2 = self._phi(2)
        phi_i = self._phi(i)
        return sp.integrate(
            2 * self._p() * (self.c1 * sp.diff(phi1, self.x) + self.c2 * sp.diff(phi2, self.x)) * sp.diff(phi_i, self.x) -
            2 * self._q() * (self.c1 * phi1 + self.c2 * phi2) * phi_i + 2 * self._f() * phi_i,
            (self.x, self.a, self.b)
        )


def main():
    n = 3
    a, b = 0, 1
    x = sp.symbols('x')
    c1, c2 = sp.symbols('c1 c2')
    ritz = Ritz(x, c1, c2, a, b, n)
    A = np.zeros((n-1, n-1))
    b = np.zeros(n-1)
    for i in range(1, n):
        expr = ritz.dPhi_dCi(i)
        coeff_c1 = expr.coeff(c1)
        coeff_c2 = expr.coeff(c2)
        const_term = expr.subs({c1: 0, c2: 0})
        A[i-1][0] = coeff_c1
        A[i-1][1] = coeff_c2
        b[i-1] = const_term

    solution = np.linalg.solve(A, b)
    print(A)
    print(b)
    answer = ritz.u().subs(c1, solution[0]).subs(c2, solution[1])
    print(answer)

    ab = np.linspace(0, 1, 100)
    my_values = [answer.subs(x, xi) for xi in ab]
    true_values = [1 / (xi ** 2 + 1) for xi in ab]
    print([abs(true_values[i] - my_values[i]) for i in range(100)])


if __name__ == '__main__':
    main()