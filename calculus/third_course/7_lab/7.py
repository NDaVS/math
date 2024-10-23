import numpy as np
from calculus.third_course.support.checker import *


def sigma(a):
    if a < 0:
        return -1

    return 1

def kd(i,j): #Kronecker delta
    if i == j:
        return 1

    return 0


def qr(A):
    n = A.shape[0]
    R = np.copy(A)
    Q = np.eye(n)

    for k in range(n - 1):
        # Шаг 1: Формируем вектор отражения ps
        ps = np.zeros(n)
        norm_x = np.sqrt(np.sum(R[k:, k] ** 2))
        ps[k] = R[k, k] + sigma(R[k, k]) * norm_x
        ps[k+1:] = R[k+1:, k]

        # Шаг 2: Формируем матрицу P_k (отражение Хаусхолдера)
        P_k = np.eye(n) - 2 * np.outer(ps, ps) / np.sum(ps[k:] ** 2)

        # Шаг 3: Применяем P_k к R и Q
        R = P_k @ R
        Q = P_k @ Q

    return Q.T, R



def get_answer(A, b):
    Q, R = qr(A)

    # Вычисляем Q^T * b
    b_tilde = np.dot(Q.T, b)

    # Инициализируем вектор x для хранения решения
    n = R.shape[1]
    x = np.zeros(n)

    # Метод обратной подстановки для решения Rx = Q^T b
    for i in range(n - 1, -1, -1):
        x[i] = (b_tilde[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    return x


def main():
    A = np.array([
        [2.2, 4, -3, 1.5, 0.6, 2, 0.7],
        [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
        [-3, 1.5, 1.8, 0.9, 3, 2, 2],
        [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
        [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
        [2, 3, 2, 3, 0.6, 2.2, 4],
        [0.7, 1, 2, 1, 0.7, 4, 3.2]
    ])

    # Вектор b - вектор свободных членов системы уравнений
    b = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7])
    x = get_answer(A, b)
    check_answer(A, b, x)


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)  # Установите ширину строки и отключите экспоненциальный формат

    main()
