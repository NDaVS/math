import numpy as np

def f(x):
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b, x)


def is_positive_definite(matrix):
    # Проверка, является ли матрица квадратной
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Проверка всех главных миноров
    for i in range(1, matrix.shape[0] + 1):
        minor = matrix[:i, :i]
        if np.linalg.det(minor) <= 0:
            return False

    return True

def gradient_descent(A, b, x0, alpha, tol=1e-6, max_iter=10000):
    if not is_positive_definite(A):
       raise ValueError("wrong matrix")

    x = x0

    for i in range(max_iter):
        grad = np.array([
            0.5 * (A[0][0] * x[0] * 2 + (A[0][1] + A[1][2]) * x[1] + (A[2, 0] + A[0, 2]) * x[2]) + b[0],
            0.5 * (A[1, 1] * x[1] * 2 + (A[0][1] + A[1][2]) * x[0] + (A[1, 0] + A[2, 1]) * x[2]) + b[1],
            0.5 * (A[2, 2] * x[2] * 2 + (A[2, 0] + A[0, 2]) * x[0] + (A[1, 0] + A[2, 1]) * x[1]) + b[2]] ) # Вычисление градиента
        x_new = x - alpha * grad  # Обновление x

        if f(x_new) > f(x):
            alpha /= 2

        if np.linalg.norm(grad) < tol:  # Условие остановки
            print(i)
            break

        # if np.linalg.norm(x_new - x) < tol:  # Условие остановки
        #     print(i)
        #     break
        x = x_new

    return x, 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b, x)  # Возврат результата


# Пример использования
A = np.array([[2, 3, 1], [2, 7, 2], [1, 3, 3]])
b = np.array([3, 4, 5])
x0 = np.array([1, 1, 1])
alpha = 0.01

x_min, f_min = gradient_descent(A, b, x0, alpha)
print(f"Минимум достигается в: {x_min}, значение функции: {f_min}")

