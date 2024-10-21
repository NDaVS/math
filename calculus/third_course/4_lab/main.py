import numpy as np


def sum_for_b(A, B, C, i, j):
    answer = 0

    for k in range(j):
        answer += B[i][k] * C[k][j]

    return A[i][j] - answer


def sum_for_c(A, B, C, i, j):
    answer = 0

    for k in range(i):
        answer += B[i][k] * C[k][j]

    return (A[i][j] - answer) / B[i][i]


def sum_for_y(B, b, y, i):
    sum_answer = 0

    for k in range(i):
        sum_answer += B[i][k] * y[k]

    return (b[i] - sum_answer) / B[i][i]


def sum_for_x(C, x, y, i):
    sum_answer = 0

    for k in range(i + 1, y.shape[0]):
        sum_answer += C[i][k] * x[k]

    return y[i] - sum_answer


def find_y(B, b):
    n = B.shape[0]
    y = np.zeros(n)

    y[0] = b[0] / B[0][0]

    for i in range(1, n):
        y[i] = sum_for_y(B, b, y, i)

    return y


def find_x(y, C, i, j):
    sum_answer = 0
    x = np.zeros(C.shape[0])

    for i in range(C.shape[0] - 1, -1, -1):
        x[i] = sum_for_x(C, x, y, i)

    x[-1] = y[-1]
    return x


def LU_decay(A, b):
    n = A.shape[0]
    B = np.zeros(A.shape)
    C = np.zeros(A.shape)

    for i in range(n):
        C[i][i] = 1
        B[i][0] = A[i][0]

        if i == 0:
            for j in range(n):
                C[0][j] = A[0][j] / B[0][0]

            continue

        for j in range(n):
            if i >= j > 0:
                B[i][j] = sum_for_b(A, B, C, i, j)

            if j > i > 0:
                C[i][j] = sum_for_c(A, B, C, i, j)

    y = find_y(B, b)

    return find_x(y, C, i, j)


def main():
    A = np.array([
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
    ])
    b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

    # A = np.array([
    #     [3, 1, -1, 2],
    #     [-5, 1, 3, -4],
    #     [2, 0, 1, -1],
    #     [1, -5, 3, -3]
    # ])
    #
    # b = np.array([6, -12, 1, 3])

    x = LU_decay(A, b)

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
        "Модуль разности произведения матрицы на  вектор ответа, полученный с помощью библиотеки np, и " +
        "вектора свободных членов: " +
        str(np.linalg.norm(A @ np.linalg.solve(A, b) - b))
    )


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)  # Установите ширину строки и отключите экспоненциальный формат

    main()
