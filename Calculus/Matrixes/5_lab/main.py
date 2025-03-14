import numpy as np
import cmath


def sum_for_b(A, B, C, i, j):
    answer = 0

    for k in range(j):
        answer += B[k][i] * B[k][j]

    return A[i][j] - answer


def sum_for_c(A, B, C, i, j):
    answer = 0

    for k in range(i):
        answer += B[i][k] * C[k][j]

    return (A[i][j] - answer) / B[i][i]


def sum_for_y(B, b, y, i):
    sum_answer = 0

    for k in range(i):
        sum_answer += B[k][i] * y[k]

    return (b[i] - sum_answer) / B[i][i]


def sum_for_x(C, x, y, i):
    sum_answer = 0

    for k in range(i + 1, y.shape[0]):
        sum_answer += C[i][k] * x[k]

    return y[i] - sum_answer


def find_y(B, b):
    n = B.shape[0]
    y = np.zeros(n, dtype=complex)

    y[0] = b[0] / B[0][0]

    for i in range(1, n):
        y[i] = sum_for_y(B, b, y, i)

    return y


def find_x(y, C):
    x = np.zeros(C.shape[0], dtype=complex)

    for i in range(C.shape[0] - 1, -1, -1):
        x[i] = sum_for_x(C, x, y, i) / C[i][i]

    return x


def square_root_method(A, b):
    n = A.shape[0]
    B = np.zeros(A.shape, dtype=complex)
    C = np.zeros(A.shape, dtype=complex)

    for i in range(n):
        if i == 0:
            B[0][0] = np.sqrt(A[0][0])

            for j in range(1, n):
                B[0][j] = A[0][j] / B[0][0]

            continue

        B[i][i] = cmath.sqrt(sum_for_b(A, B, C, i, i))

        for j in range(i + 1, n):
            B[i][j] = sum_for_b(A, B, C, i, j) / B[i][i]

    y = find_y(B, b)
    x = find_x(y, B)

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

    x = square_root_method(A, b)

    print("Вектор ответа: " + str(x))

    print(
        "Модуль разности нашего решения и решения через библиотеку np: " +
        str(np.linalg.norm(np.linalg.solve(A, b) - x))
    )

    print(
        "Модуль разности произведения матрицы на наш вектор ответа и вектора свободных членов: " +
        str(np.linalg.norm(A @ x - b))
    )

    print(
        "Модуль разности произведения матрицы на вектор ответа, полученный с помощью библиотеки np, и " +
        "вектора свободных членов: " +
        str(np.linalg.norm(A @ np.linalg.solve(A, b) - b))
    )


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)  # Установите ширину строки и отключите экспоненциальный формат

    main()
