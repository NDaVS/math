import numpy as np

from calcus_math.pract.support.monotonous_running import tridiagonal_solver, checker


def TDMA(a, b, c, f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, b, c, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0] * n

    for i in range(1, n):
        alpha.append(-b[i] / (a[i] * alpha[i - 1] + c[i]))
        beta.append((f[i] - a[i] * beta[i - 1]) / (a[i] * alpha[i - 1] + c[i]))

    x[n - 1] = beta[n - 1]

    for i in range(n - 1, -1, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

    return x


def main():
    # A = np.array([
    #     [2, -1, 0],
    #     [-1, 2, -1],
    #     [0, -1, 2]
    # ])
    # b = np.array([1, 2, 3])

    # A = np.array([
    #     [3, 6, 0],
    #     [1, 12, 2],
    #     [0, 3, 2]
    # ])
    # b = np.array([4, 7, 12])

    # A = np.array([
    #     [1, 2, 0],
    #     [2, 1, 3],
    #     [4, 1, 2],
    #     [0, 2, 1]
    # ])
    # b = np.array([4, 2, 3, 5])

    A = np.array([
    [4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    [7.0, 5.0, -2.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 10.0, -10.0, 0.0, 0.0],
    [0.0, 0.0, -5.0, 4.0, 9.0, 0.0],
    [0.0, 0.0, 0.0, -2.0, 6.0, -4.0],
    [0.0, 0.0, 0.0, -1.0, -7.0, 1.0],

])
    b = np.array([9, 3, 1, 2, 7, -1.6])

    x = tridiagonal_solver(A, b)
    print(sum(x))
    print(checker(A, b, x))


def dev():
    # Входные данные
    a = [1.0, 2.0, 3.0, 4.0]
    b = [-2.0, -3.0, -4.0, -5.0]
    c = [1.0, 2.0, 3.0, 4.0]
    f = [1.0, 2.0, 3.0, 4.0]

    # Вызов функции TDMA
    x = TDMA(a, b, c, f)

    # Вывод результата
    print(x)


if __name__ == "__main__":
    main()
    # dev()
