import math
from sympy import symbols, diff, log


def function(x):
    return x ** 2 + math.log(x, math.e) - 4


def values(a, b, step):
    table = []
    while a <= b:
        table.append((a, function(a)))
        a += step
    return table


def L1(x, x_i, x_i1):
    return function(x_i) * (x - x_i1) / (x_i - x_i1) + function(x_i1) * (x - x_i) / (x_i1 - x_i)


def w(x, x_i, x_i1):
    return (x - x_i) * (x - x_i1)


def R(ddf, w):
    return ddf * w/2


def main():
    # Строим таблицу значений (таблично заданная функция)#
    a = 1.5
    b = 2.0
    step = (b - a) / 10
    x_asterX = 1.52
    table = values(a, b, step)

    # Ищем X_i и X_{i+1}#
    index = 0  # Наш X_i
    while x_asterX < table[index][0]:
        index += 1

    index1 = index + 1

    x = symbols('x')
    f = x ** 2 + log(x) - 4
    df = diff(f, x)  # Первая производная
    ddf = diff(df, x)  # Вторая производная
    dddf = diff(ddf, x)  # Третья производная

    l1 = L1(x_asterX, table[index][0], table[index1][0])

    value = function(x_asterX)  # y(x*)
    R_1 = l1 - value
    R_min = ddf(table[index][0]) * w(x_asterX, table[index][0], table[index1][0])/ 2
    R_max = R(ddf(table[index1][0]), w(x_asterX, table[index][0], table[index1][0]))
    print(R_min)
    print(R_max)
    # print(R_1)
    # print(*table, sep='\n')


if __name__ == '__main__':
    main()
