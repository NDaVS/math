from calculus.labs.support.interpolations import lagrange, table, get_corner, newton
from calculus.labs.support.subs import omega, dnf
from calculus.labs.support.checker import check_results
import sympy as sp
import math


# made by Lunk, feat IDrumo



def main():
    # Определение переменной и функции
    x = sp.symbols('x')
    f = (x - 1) ** 2 - sp.exp(-x)
    a, b = 1.0, 1.5
    n = 10
    h = (b - a) / n

    # Точки для интерполяции
    x_star = 1.07
    x_double_star = 1.02
    x_triple_star = 1.47
    x_quadruple_star = 1.27

    # Заданные точки для интерполяции
    all_x = [a + i * h for i in range(n + 1)]
    all_y = [f.subs(x, xi) for xi in all_x]

    lagrange_function = lagrange(all_x, all_y, x)
    lagrange_df = dnf(lagrange_function, x, 1)
    df = dnf(f, x, 1)
    # paint_function(lagrange_df, a, b)
    # paint_function(df, a, b)



    # check_results(f, lagrange_function, x, a, all_x[-1], 10, a)
    a= sp.symbols('a')
    b= sp.symbols('b')
    c= sp.symbols('c')
    d= sp.symbols('d')
    f = (x-a)*(x-b)*(x-c)
    new_function = sp.diff(f, x)
    print(new_function)
    print(new_function.simplify())
    print(sp.simplify(new_function))
if __name__ == "__main__":
    main()
