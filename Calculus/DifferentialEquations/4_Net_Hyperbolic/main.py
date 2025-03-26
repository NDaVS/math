import matplotlib.pyplot as plt
import numpy as np


def g(x, t):
    return 0  # Внешнее воздействие


def phi(x):
    return 0  # Начальная скорость


def psi(x):
    return np.sin(np.pi * x)  # Начальное условие


def gamma_0(t):
    return t  # Граничное условие слева


def gamma_1(t):
    return -2 * t ** 2  # Граничное условие справа


def plot_solution(u, h, tau):
    N, M = u.shape
    x = np.linspace(0, (M - 1) * h, M)
    t = np.linspace(0, (N - 1) * tau, N)

    plt.figure(figsize=(10, 6))
    for i in range(0, N, max(1, N // 10)):
        plt.plot(x, u[i], label=f't = {t[i]:.2f}')

    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.title('Solution')
    plt.grid()
    plt.show()


def main():
    M, N = 100, 1000  # Количество узлов по x и t
    l, T = 1, 1  # Длина стержня и время
    h, tau = l / (M - 1), T / (N - 1)
    a = 3  # Скорость волны

    u = np.zeros((N, M))

    # Начальное условие
    for j in range(M):
        u[0, j] = psi(j * h)

    # Граничные условия
    for i in range(N):
        u[i, 0] = gamma_0(i * tau)
        u[i, -1] = gamma_1(i * tau)

    # Инициализация второго временного слоя
    for j in range(1, M - 1):
        u[1, j] = u[0, j] + tau * phi(j * h)

    # Основная разностная схема (явная схема)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            u[i + 1, j] = (
                a ** 2 * tau ** 2 / h ** 2 * (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1])
                + 2 * u[i, j] - u[i - 1, j]
                + tau ** 2 * g(j * h, i * tau)
            )

    plot_solution(u, h, tau)


if __name__ == '__main__':
    main()
