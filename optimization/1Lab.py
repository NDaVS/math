import numpy as np
import math


# Function definition
def f(x):
    return (1 / 4) * x ** 4 + x ** 2 - 8 * x + 12


# Функция для генерации последовательности Фибоначчи
def fibonacci_sequence(n):
    fib_seq = [1, 1]
    for i in range(2, n):
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    return fib_seq


def fibonacci_number(n):
    return (
            math.pow((1 + math.sqrt(5)) / 2, n) -
            math.pow((1 - math.sqrt(5)) / 2, n)
    ) / math.sqrt(5)


# Метод Фибоначчи для поиска минимума
def fibonacci_method(f, a, b, epsilon=1e-5):
    # Определим количество шагов (N), необходимых для достижения точности
    N = 1
    while (b - a) / epsilon > fibonacci_number(N):
        N += 1

    # Получаем последовательность Фибоначчи
    # fib_seq = fibonacci_sequence(N)

    # Начальные точки
    x1 = a + (fibonacci_number(N - 2) / fibonacci_number(N)) * (b - a)
    x2 = a + (fibonacci_number(N - 1) / fibonacci_number(N)) * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    for k in range(1, N - 1):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fibonacci_number(N - k - 2) / fibonacci_number(N - k)) * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fibonacci_number(N - k - 1) / fibonacci_number(N - k)) * (b - a)
            f2 = f(x2)

    # Находим финальный минимум
    return (a + b) / 2, N


def dichotomy_method(f, a, b, epsilon=1e-5, delta=1e-6):
    iterations = 0

    while (b - a) > epsilon:
        # Выбираем две точки, симметричные относительно середины интервала и на расстоянии друг от друга 2 delta
        x1 = (a + b) / 2 - delta
        x2 = (a + b) / 2 + delta

        # Вычисляем значения функции в точках
        f1 = f(x1)
        f2 = f(x2)

        # Сравниваем значения и выбираем новый интервал
        if f1 < f2:
            b = x2
        else:
            a = x1

        iterations += 1

    # Возвращаем приближенное значение минимума
    return (a + b) / 2, iterations


# Given parameters
a = 0
b = 2
epsilon = 0.005

# Ищем минимум функции на интервале [0, 2]
minimum, dichotomy_interations = dichotomy_method(f, 0, 2, epsilon)
print(f"Минимум функции находится в точке: {minimum}")
print(dichotomy_interations)

minimum, fibonacci_interations = fibonacci_method(f, 0, 2, epsilon)
print(f"Минимум функции находится в точке: {minimum}")
print(fibonacci_interations)
