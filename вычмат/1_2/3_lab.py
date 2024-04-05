import sympy as sp
import math

def function(x):
    # Используем sp.log для логарифма по основанию e
    return x ** 2 + sp.log(x) - 4

def values(a, b, step):
    table = []
    x = a
    while x <= b:
        table.append((x, function(x)))
        x += step
    return table

def lagrange_polynomial(points, x):
    L = 0
    for i, (xi, yi) in enumerate(points):
        Li = 1
        for j, (xj, _) in enumerate(points):
            if i != j:
                Li *= (x - xj) / (xi - xj)
        L += yi * Li
    return L
def take_dif(func, x, n):
    new_fun = func
    for _ in range(n):
        new_fun = sp.diff(new_fun, x)
    return new_fun
def omega(a, b, step, x):
    result = 1
    while round(a, 2) <= b:
        result *= (x - a)
        a += step
    return result
def main():
    # Определяем символы
    x = sp.symbols('x')
    n=10
    k=1
    m=3
    a = 1.5
    b = 2.0
    step = (b - a) / 10

    points = values(a, b, step)

    L = lagrange_polynomial(points, x)

    print(f"Многочлен Лагранжа: {L}")

    L_diff = sp.diff(L, x)

    # Создаем функцию f с использованием символов SymPy для последующего дифференцирования
    f = x ** 2 + sp.log(x) - 4
    d = take_dif(f, x, n)
    df = take_dif(f, x, n)
    R_1 = d.subs(x, 1.5) - L_diff.subs(x, 1.5)
    r_min = (df.subs(x, a) / math.factorial(11)) * omega(a, b, step, x)

    r_max = df.subs(x, b) / math.factorial(11) * omega(a, b, step, x)

    print(f"Производная многочлена Лагранжа: {L_diff}")

    # Вычисляем значение многочлена Лагранжа и его производной в точке x = 1.5
    print(L.subs(x, 1.5))
    # Используем функцию evalf для вычисления значения функции, так как function теперь возвращает символьное выражение
    print(function(1.5).evalf())
    print('=============================')
    print(L_diff.subs(x, 1.5))
    print(d.subs(x, 1.5))
    print('=============================')
    print(R_1)
    print(r_min.subs(x,a))
    print(r_max.subs(x,b))

if __name__ == '__main__':
    main()
