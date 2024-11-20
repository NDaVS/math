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


def simple_iteration_method(A, b, x0, tol=1e-5, max_iter=1000):
    for i in range(b.shape[0]):
        delimiter = A[i, i]
        A[i] = A[i] / delimiter
        b[i] /= delimiter

    if check_matrix(A):
        for k in range(max_iter):
            new_x = np.zeros_like(b)

            for i in range(b.shape[0]):
                tmp = 0

                for j in range(b.shape[0]):
                    if (i != j):
                        tmp -= A[i][j] * x0[j]

                new_x[i] = tmp + b[i]

            if (np.linalg.norm(new_x - x0, ord=np.inf)) < tol:
                return new_x, k

            x0 = new_x

        raise ValueError('error')


def relaxation_method(A, b, x0, omega, epsilon=1e-5, max_iterations=10000000):
    n = len(b)
    x = x0.copy()

    for k in range(max_iterations):
        x_old = x.copy()

        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i, i]

            # Overflow check
            if np.isinf(x[i]) or np.isnan(x[i]):
                raise ValueError("Overflow or NaN detected. Check matrix conditioning or reduce omega.")

        # Convergence check
        if np.linalg.norm(x - x_old, ord=np.inf) < epsilon:
            return x, k + 1

    raise ValueError(
        f"Convergence not achieved after {max_iterations} iterations with omega={omega}, epsilon={epsilon}")


def main():
    A = np.array([
        [10.9, 1.2, 2.1, 0.9],
        [1.2, 11.2, 1.5, 2.5],
        [2.1, 1.5, 9.8, 1.3],
        [0.9, 2.5, 1.3, 12.1],
    ])
    b = np.array([-7.0, 5.3, 10.3, 24.6])

    A1 = np.array([
        [3.82, 1.02, 0.75, 0.81],
        [1.05, 4.53, 0.98, 1.53],
        [0.73, 0.85, 4.71, 0.81],
        [0.88, 0.81, 1.28, 3.50]
    ], dtype=float)

    b1 = np.array([15.655, 22.705, 23.480, 16.110], dtype=float)

    x0 = np.ones(b.shape[0])

    omegas = [0.01, 0.5, 1, 1.5, 1.99]

    print('Исходная матрица:')
    print(A)

    print("Вектор свободных членов", b)

    print('\nРезультаты для метода простой итерации:')
    x, i = simple_iteration_method(A, b, x0, )
    print(f'Сходимость метода за {i} итераций')

    check_answer(A, b, x)

    for omega in omegas:
        print(f'\nРезультаты для метода релаксации с параметром omega = {omega}:')
        x, iterations = relaxation_method(A, b, x0, omega)
        print(f'Сходимость метода за {iterations} итераций')
        check_answer(A, b, x)

    print('\n\nИсходная матрица:')
    print(A1)

    print("Вектор свободных членов", b1)

    print('\nРезультаты для метода простой итерации:')
    x, i = simple_iteration_method(A1, b1, x0)
    print(f'Сходимость метода за {i} итераций')
    check_answer(A1, b1, x)

    for omega in omegas:
        try:
            print(f'\nРезультаты для метода релаксации с параметром omega = {omega}:')
            x, iterations = relaxation_method(A1, b1, x0, omega)
            print(f'Сходимость метода за {iterations} итераций')
            check_answer(A1, b1, x)
        except Exception as e:
            print(f"Convergence not achieved after {10}M iterations with omega={omega}, epsilon={1e-3}")



if __name__ == '__main__':
    np.set_printoptions(linewidth=200, suppress=True)  # Установите ширину строки и отключите экспоненциальный формат

    main()
