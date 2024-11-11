from calculus.third_course.support.checker import *


def check_matrix(A):
    isValid = True

    for i in range(A.shape[0]):
        tmp = 0
        for j in range(A.shape[0]):
            if (i != j):
                tmp += np.abs(A[i, j])
        if abs(A[i, i]) < tmp:
            isValid = False
            return isValid

    return isValid


def simple_iteration_method(A, b, x0, tol=1e-6, max_iter=1000):
    for i in range(b.shape[0]):
        delimiter = A[i, i]
        A[i] = A[i] / delimiter
        b[i] /= delimiter

    if check_matrix(A):
        for _ in range(max_iter):
            new_x = np.zeros_like(b)

            for i in range(b.shape[0]):
                tmp = 0

                for j in range(b.shape[0]):
                    if (i != j):
                        tmp -= A[i][j] * x0[j]

                new_x[i] = tmp + b[i]

            if (np.linalg.norm(new_x - x0, ord=np.inf)) < tol:
                return new_x

            x0 = new_x

        raise ValueError('error')


def first_sum(A, x, i):
    return np.dot(A[i, :i], x[:i])  # Сумма до i-го элемента


def second_sum(A, x, i, n):
    return np.dot(A[i, i + 1:n], x[i + 1:n])  # Сумма после i-го элемента


def relaxation_method(A, b, x0, omega, epsilon=1e-3, max_iterations=1000000):
    n = len(b)
    x = x0.copy()

    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            x_new[i] = (1 - omega) * x[i] + (b[i] - first_sum(A, x, i) - second_sum(A, x, i, n)) * omega / A[i, i]

        if np.linalg.norm(x_new - x) < epsilon:
            return x_new

        x = x_new

    raise ValueError('Слишком много итераций')


def main():
    # A = np.array([
    #     [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
    #     [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
    #     [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
    #     [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
    #     [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
    #     [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
    #     [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
    # ])
    # b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

    A = np.array([
        [10, 2, 1],
        [1, 10, 2],
        [1, 1, 10]
    ], dtype=float)

    b = np.array([10, 12, 8], dtype=float)
    x0 = np.ones(b.shape[0])
    # x = simple_iteration_method(A, b, x0, 1e-15)
    x = relaxation_method(A, b, x0, 1)

    check_answer(A, b, x)
    # print(x)


if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)  # Установите ширину строки и отключите экспоненциальный формат

    main()
