import math


def do_sin(x, n):

    return math.pow(-1, n) * math.pow(x, 2 * n + 1) / math.factorial(2 * n + 1)


def sin(degree, accuracy):
    answer = 0
    n = 1
    x = degree * 3.1415926535 / 180
    while do_sin(x, n) >= accuracy:
        answer += do_sin(x, n)
        n += 1
    print(answer)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sin(0.5236, 10 ^ (-4))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
