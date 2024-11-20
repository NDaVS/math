from calculus.third_course.support.checker import *


def mu(r, w):
    return np.dot(r, w @ w.T @ r) / np.dot(w @ w.T @ r, w @ w.T @ r)


def r(A, b, x):
    return np.dot(A, x) - b


def gradient_descent(A, b, tol=1e-5, max_iterations=1000000):
    x = np.zeros_like(b)
    for i in range(max_iterations):
        x_old = x.copy()
        r_k = r(A, b, x)
        mu_k = mu(r_k, A)
        x = x_old - mu_k * A.T @ r_k

        if np.linalg.norm(x - x_old) < tol:
            return x, i

    raise ValueError(
        f"Convergence not achieved after {max_iterations} iterations")


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

    x, i = gradient_descent(A, b)
    print(f'Итерационный процесс завершился за {i} итераций')
    check_answer(A, b, x)


    x, i = gradient_descent(A1, b1)
    print(f'\n\nИтерационный процесс завершился за {i} итераций')
    check_answer(A1, b1, x)

if __name__ == '__main__':
    np.set_printoptions(linewidth=200,
                        suppress=True)  # Установите ширину строки и отключите экспоненциальный формат

    main()
