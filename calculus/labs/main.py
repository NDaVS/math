import math


def do_sin(x, n):
    return (-1) ** (n - 1) * x ** (2 * n - 1) / math.factorial(2 * n - 1)


def sin(degree, accuracy):
    answer = 0
    n = 1
    x = degree
    if x >= 2 * math.pi:
        x %= 2 * math.pi
    while abs(do_sin(x, n)) >= accuracy:
        answer += do_sin(x, n)
        n += 1
    print('=' * 60)
    print(f"Приближённое значение для sin({degree})≈{answer}\n"
          f"Точность вычислений: {accuracy}\n"
          f"Точное значение для sin({degree})= {math.sin(x)}\n"
          f"Погрешность: {abs(answer - math.sin(x))}\n"
          f"Погрешность{'не' if abs(answer - math.sin(x)) > accuracy else ''} удовлетворяет выбранной точности")


if __name__ == '__main__':
    sin(0.5236, 10 ** (-4))
    sin(52.36, 10 ** (-8))
    print('=' * 60)
