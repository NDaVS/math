from calculus.labs.support.interpolations import lagrange, table, get_corner, newton
from calculus.labs.support.subs import omega, dnf
from calculus.labs.support.checker import check_results
import sympy as sp
import math


def main():
    x = sp.symbols('x')
    f = x / 2 - sp.cos(2/x)
    a, b = 0.4, 0.9
    n = 10
    h = round((b - a) / 10, 2)
    all_x = [a + i * h for i in range(4)]
    all_y = [f.subs(x, xi) for xi in all_x]
    new_f = lagrange(all_x, all_y, x)
    new_df = dnf(new_f, x, 2)
    df = dnf(f, x, 2)
    print(new_df.subs(x, a+4*h))
    print(df.subs(x, a+3*h))
    check_results(df, new_df, x, a, all_x[-1], 3, a+4*h)


if __name__ == '__main__':
    main()
