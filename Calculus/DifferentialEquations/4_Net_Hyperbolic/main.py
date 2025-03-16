import matplotlib.pyplot as plt
import numpy as np


def g(x, t):
    return 0


def phi(x):
    return 0


def psi(x):
    return 0


def gamma_0(t):
    return t


def gamma_1(t):
    return -2 * t ** 2


def plot_solution(u, h, tau):
    M, N = u.shape
    x = np.linspace(0, (M - 1) * h, M)
    t = np.linspace(0, (N - 1) * tau, N)

    plt.figure(figsize=(10, 6))
    # for i in range(0, N, max(1, N // 10)):
    #     plt.plot(x, u[i], label=f't = {t[i]:.2f}')
    #
    #     plt.xlabel('x')
    #     plt.ylabel('u')
    #     plt.legend()
    #     plt.title('Solution')
    #     plt.grid()
    #     plt.show()
    for i in range(0, N, max(1, N // 10)):
        plt.plot(x, u[i], label=f't = {t[i]:.2f}')

    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.title('Solution')
    plt.grid()
    plt.show()


def main():
    M, N = 10, 10
    l, T = 1, 1
    h, tau = l / M, T / N
    a = 3

    u = np.zeros((M, N))

    for i in range(M):
        u[0, i] = gamma_0(i * h)
        u[N - 1, i] = gamma_1(i * h)

    for i in range(1, M - 1):
        u[1, i] = u[0, i] + tau * phi(i * h)

    for i in range(2, N - 1):
        for j in range(1, M - 1):
            u[i, j] = a ** 2 * tau ** 2 / h ** 2 * (u[i - 1, j + 1] - 2 * u[i - 1, j] + u[i - 1, j - 1]) + 2 * u[
                i - 1, j] - u[i - 2, j] + tau ** 2 * g(j * h, i * tau)
    plot_solution(u, h, tau)


if __name__ == '__main__':
    main()
