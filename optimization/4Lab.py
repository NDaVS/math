import numpy as np


class Gradient:

    def __init__(self, A: np.array, b: np.array, x0: np.array, alpha: float, h: float, max_iter=100000):
        self.max_iter = max_iter
        self.x = None
        self.alpha = alpha
        self.A = A
        self.b = b
        self.x0 = x0
        self.h = [h] * 3

    def f(self, x):
        return 0.5 * np.dot(x.T, np.dot(self.A, x)) + np.dot(self.b, x)

    def is_positive_definite(self, matrix):
        # Проверка, является ли матрица квадратной
        if matrix.shape[0] != matrix.shape[1]:
            return False

        # Проверка всех главных миноров
        for i in range(1, matrix.shape[0] + 1):
            minor = matrix[:i, :i]
            if np.linalg.det(minor) <= 0:
                return False

        return True

    def gradient_descent(self, tol=1e-6, max_iter=10000):
        if not self.is_positive_definite(self.A):
            raise ValueError("wrong matrix")

        x = self.x0

        for i in range(max_iter):
            grad = np.array([
                0.5 * (self.A[0][0] * x[0] * 2 + (self.A[0][1] + self.A[1][2]) * x[1] + (self.A[2, 0] + self.A[0, 2]) *
                       x[2]) + self.b[0],
                0.5 * (self.A[1, 1] * x[1] * 2 + (self.A[0][1] + self.A[1][2]) * x[0] + (self.A[1, 0] + self.A[2, 1]) *
                       x[2]) + self.b[1],
                0.5 * (self.A[2, 2] * x[2] * 2 + (self.A[2, 0] + self.A[0, 2]) * x[0] + (self.A[1, 0] + self.A[2, 1]) *
                       x[1]) + self.b[2]])  # Вычисление градиента
            x_new = x - self.alpha * grad  # Обновление x

            if self.f(x_new) > self.f(x):
                self.alpha /= 2

            if np.linalg.norm(grad) < tol:  # Условие остановки
                print(i)
                break

            x = x_new

        return x, 0.5 * np.dot(x.T, np.dot(self.A, x)) + np.dot(self.b, x)  # Возврат результата

    def check_h(self, x: np.array, i: int):
        old_f = self.f(x)
        local_h = self.h

        while True:

            x[i] = x[i] + local_h[i]

            if old_f > self.f(x):
                self.x = x
                return

            x[i] -= 2 * local_h[i]

            if old_f > self.f(x):
                self.x = x
                return

            x[i] += local_h[i]

            local_h[i] /= 2

    def partition_descent(self):
        x = self.x0

        for i in range(self.max_iter):
            old_f = self.f(x)
            for i in range(len(x)):
                self.check_h(x, i)
            if np.abs(old_f - self.f(self.x)) < self.alpha:
                return self.x, self.f(self.x)


def main():
    # Пример использования
    A = np.array([[2, 3, 1],
                  [2, 7, 2],
                  [1, 3, 3]])
    b = np.array([3, 4, 5])
    x0 = np.array([1, 1, 0], dtype='float')
    alpha = 0.01
    h = 0.2
    gradient = Gradient(A, b, x0, alpha, h)

    x_min, f_min = gradient.gradient_descent()
    print(f"Обычный градиент.\nМинимум достигается в: {x_min}, значение функции: {f_min}\n\n")

    x_min, f_min = gradient.partition_descent()
    print(f"Метод поэлементного спуска.\nМинимум достигается в: {x_min}, значение функции: {f_min}\n\n")


if __name__ == '__main__':
    main()
