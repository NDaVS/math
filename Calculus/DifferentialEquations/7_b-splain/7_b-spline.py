import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def p(x):
    return - (1 + x)


def q(x):
    return -1


def f(x):
    return 2/((x + 1) ** 3)


def A(h, x):
    return 1 / (3 * h) * (1 - 0.5 * p(x) * h + 1 / 6 * q(x) * h ** 2)


def D(h, x):
    return 1 / (3 * h) * (1 + 0.5 * p(x) * h + 1 / 6 * q(x) * h ** 2)


def C(h, x):
    return -A(h, x) - D(h, x) + 1 / 6 * q(x) * h * 2


def F(h):
    return 1 / 6 * f(2 * h)


def A_m1(alpha_1, beta_1, h):
    return alpha_1 * h - 3 * beta_1


def C_m1(alpha_1, h):
    return 2 * alpha_1 * 2 * h


def D_m1(alpha_1, beta_1, h):
    return alpha_1 * h + 3 * beta_1


def F_m1(gamma_1, h):
    return 2 * gamma_1 * 3 * h


def tilda_C_0(alpha, beta, x_0, h):
    return C(h, x_0) - C_m1(alpha, h) * A(h, x_0) / A_m1(alpha, beta, h)


def tilda_D_0(alpha, beta, x_0, h):
    return D(h, x_0) - D_m1(alpha, beta, h) * A(h, x_0) / A_m1(alpha, beta, h)


def tilda_F_0(alpha, beta, gamma, x_0, h):
    return F(h) - F_m1(gamma, h) * A(h, x_0) / A_m1(alpha, beta, h)


def tilda_A_n(alpha, beta, x_n, h):
    return A(h, x_n) - A_m1(alpha, beta, h) * D(h, x_n) / D_m1(alpha, beta, h)


def tilda_C_n(alpha, beta, x_0, h):
    return C(h, x_0) - C_m1(alpha, h) * D(h, x_0) / D_m1(alpha, beta, h)


def tilda_F_n(alpha, beta, gamma, x_n, h):
    return F(h) - F_m1(gamma, h) * D(h, x_n) / D_m1(alpha, beta, h)


def B(i, x, ab, h):
    k = 1 / 6
    t = (x - ab[i]) / h
    if x < ab[i - 2]:
        return 0
    if ab[i - 2] <= x < ab[i - 1]:
        return k * t ** 3
    if ab[i - 1] <= x < ab[i]:
        return k * (1 + 3 * t + 3 * t ** 2)
    if ab[i] <= x < ab[i + 1]:
        return k * (1 + 3 * (1 - t) + 3 * t * (1 - t) ** 2)
    if ab[i + 1] <= x < ab[i + 1]:
        return k * (1 - t) ** 3
    return 0


def S(i, b):
    return b[i] * 1 / 6 + b[i + 1] * 2 / 3 + 1 / 6 * b[i + 2]


def get_b(alpha_1, beta_1, alpha_2, beta_2, gamma_1, gamma_2, x, h, n):
    A_matrix = np.zeros((n + 1, n + 1))
    B_vector = np.zeros(n + 1)

    A_matrix[0][0] = tilda_C_0(alpha_1, beta_1, x[0], h)
    A_matrix[0][1] = tilda_D_0(alpha_1, beta_1, x[0], h)
    B_vector[0] = tilda_F_0(alpha_1, beta_1, gamma_1, x[0], h)

    A_matrix[-1][-1] = tilda_C_n(alpha_2, beta_2, x[-1], h)
    A_matrix[-1][-2] = tilda_A_n(alpha_2, beta_2, x[-1], h)
    B_vector[-1] = tilda_F_n(alpha_2, beta_2, gamma_2, x[-1], h)

    for i in range(1, n):
        A_matrix[i][i - 1] = A(h, x[i])
        A_matrix[i][i] = C(h, x[i])
        A_matrix[i][i + 1] = D(h, x[i])
        B_vector[i] = F(h)

    return np.linalg.solve(A_matrix, B_vector)

def main():
    alpha_1, beta_1 = 1, 0
    alpha_2, beta_2 = 1, 0
    gamma_1, gamma_2 = 1,0.5
    a, b = 0, 1
    n = 3
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)

    b = get_b(alpha_1, beta_1, alpha_2, beta_2, gamma_1, gamma_2, x, h, n)

    b_m1 = (F_m1(gamma_1, h) - b[0] * C_m1(alpha_1, h) - b[1] * D_m1(alpha_1, beta_1, h)) / A_m1(alpha_1, beta_1, h)
    b_np1 = (F_m1(gamma_2, h) - b[-2] * A_m1(alpha_2, beta_2, h) - b[-1] * C_m1(alpha_2, h)) / D_m1(alpha_2, beta_2, h)
    b = np.append([b_m1], np.append(b, b_np1))
    my_values = []
    for i in range(n + 1):
        my_values.append(S(i, b))
    true_values = [1/(xi + 1) for xi in x]
    errors = [abs(true_values[i] - my_values[i]) for i in range(n + 1)]

    table = PrettyTable()
    table.field_names = ["x", "Истинное значение", "B-Сплайны", "Ошибка"]

    for i in range(n  + 1):
        table.add_row([f"{x[i]:.8f}", f"{true_values[i]:.12f}", f"{my_values[i]:.12f}", f"{errors[i]:.12f}"])

    print(table)

    plt.figure(figsize=(12,8))
    plt.plot(x, my_values, label='B-Сплайн', color='blue')
    plt.plot(x, true_values, label='Истинное значение', color='red')
    plt.title('B-Сплайн')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
