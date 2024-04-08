from calculus.labs.support.interpolations import lagrange, table, get_corner, newton
from calculus.labs.support.subs import omega, dnf
from calculus.labs.support.checker import check_results
import sympy as sp
import math

# made by Lunk, feat IDrumo

def main():
    k, m, n, = 2, 4, 4  # parameters
    a, b = 0.4, 0.9
    n_dots = 10

    x = sp.symbols('x')  # original function
    f = x / 2 - sp.cos(2 / x)

    h = round((b - a) / n_dots, 2)  # step

    all_x = [a + i * h for i in range(n)]  # array of xs
    all_y = [f.subs(x, xi) for xi in all_x]  # array of ys

    new_f = lagrange(all_x, all_y, x)  # lagrange function

    new_df = dnf(new_f, x, k)  # original function differentiate
    df = dnf(f, x, k)  # lagrange function differentiate

    check_results(df, new_df, x, all_x[0], all_x[-1], n, a + m * h)  # result


if __name__ == '__main__':
    main()
