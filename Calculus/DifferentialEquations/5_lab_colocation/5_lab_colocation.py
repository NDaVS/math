import numpy as np
import sympy as sp
from scipy.integrate import quad

# Определяем символическую переменную
x = sp.symbols('x')

# Определяем базисные функции (удовлетворяющие краевым условиям)
def basis_functions(n):
    return [x**i * (1 - x) for i in range(1, n+1)]

# Определяем параметры метода Ритца
N = 2  # Количество базисных функций
phi = basis_functions(N)
c = sp.symbols(f'c1:{N+1}')  # Коэффициенты метода Ритца

# Аппроксимация решения
u_N = sum(c[i] * phi[i] for i in range(N))

# Определение уравнения для метода Ритца
u_prime = sp.diff(u_N, x)
u_double_prime = sp.diff(u_prime, x)

# Определяем функцию веса
V_x = 4*x/(x**2 + 1)
F_x = -3/(x**2 + 1)**2

# Функционал метода Ритца
residual = u_double_prime + V_x * u_prime - u_N - F_x

# Вычисляем систему уравнений (Галёркин)
equations = [sp.integrate(residual * phi[i], (x, 0, 1)) for i in range(N)]

# Решаем систему уравнений
solution = sp.solve(equations, c)

# Подставляем найденные коэффициенты в аппроксимацию
u_N_solution = u_N.subs(solution)

# Выводим найденное приближенное решение
print("Приближенное решение методом Ритца:")
print(sp.simplify(u_N_solution))
