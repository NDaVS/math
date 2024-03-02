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


def L2(x, x_im1, x_i, x_i1):
    return function(x_im1) * (x - x_i) * (x - x_i1) / ((x_im1 - x_i) * (x_im1 - x_i1)) + function(x_i) * (
            x - x_im1) * (x - x_i1) / ((x_i - x_im1) * (x_i - x_i1)) + function(x_i1) * (
            x - x_im1) * (x - x_i) / ((x_i1 - x_im1) * (x_i1 - x_i))


def w2(x, x_i, x_i1):
    return (x - x_i) * (x - x_i1)


def w3(x, x_im1, x_i, x_i1):
    return (x - x_im1) * (x - x_i) * (x - x_i1)


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

    x_i = table[index][0]
    x_i1 = table[index1][0]
    x_im1 = a - step

    x = symbols('x')
    f = x ** 2 + log(x) - 4
    df = diff(f, x)  # Первая производная
    ddf = diff(df, x)  # Вторая производная
    dddf = diff(ddf, x)  # Третья производная

    l1 = L1(x_asterX, x_i, x_i1)

    value = function(x_asterX)  # y(x*)
    R_1 = value - l1
    R1_min = ddf.subs(x, x_i) * w2(x_asterX, x_i, x_i1) / 2
    R1_max = ddf.subs(x, x_i1) * w2(x_asterX, x_i, x_i1) / 2

    if R1_max < R1_min:
        R1_max, R1_min = R1_min, R1_max

    if R1_min < R_1 < R1_max:
        print("Greate! First inequality is correct")
    else:
        print(f"Bad news about first inequality, it`s wrong: {R1_min}, {R_1}, {R1_max}")
    print("Round 2. Fight!")

    l2 = L2(x_asterX, x_im1, x_i, x_i1)

    R2_min = dddf.subs(x, x_i) * w3(x_asterX, x_im1, x_i, x_i1) / 6
    R2_max = dddf.subs(x, x_i1) * w3(x_asterX, x_im1, x_i, x_i1) / 6

    R_2 = function(x_asterX) - l2
    if R2_min < R_2 < R2_max:
        print("Greate! Second inequality is correct")
    else:
        print(f"Bad news about second inequality, it`s wrong: {R2_min}, {R_2}, {R2_max}")


if __name__ == '__main__':
    main()
