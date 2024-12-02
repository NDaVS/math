import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Основной класс Gradient из предоставленного кода
class Gradient:

    def __init__(self, A: np.array, b: np.array, x0: np.array, alpha: float, h: float, max_iter=100000):
        self.max_iter = max_iter
        self.x = None
        self.alpha = alpha
        self.A = A
        self.b = b
        self.x0 = x0
        self.h = [h] * 3
        self.trajectory = []  # Для записи траектории

    def f(self, x):
        return 0.5 * np.dot(x.T, np.dot(self.A, x)) + np.dot(self.b, x)

    def is_positive_definite(self, matrix):
        if matrix.shape[0] != matrix.shape[1]:
            return False
        for i in range(1, matrix.shape[0] + 1):
            minor = matrix[:i, :i]
            if np.linalg.det(minor) <= 0:
                return False
        return True

    def gradient_descent(self, tol=1e-6, max_iter=10000):
        if not self.is_positive_definite(self.A):
            raise ValueError("Matrix is not positive definite")

        x = self.x0
        self.trajectory = [x.copy()]

        for i in range(max_iter):
            grad = np.array([
                0.5 * (self.A[0][0] * x[0] * 2 + (self.A[0][1] + self.A[1][2]) * x[1] + (self.A[2, 0] + self.A[0, 2]) *
                       x[2]) + self.b[0],
                0.5 * (self.A[1, 1] * x[1] * 2 + (self.A[0][1] + self.A[1][2]) * x[0] + (self.A[1, 0] + self.A[2, 1]) *
                       x[2]) + self.b[1],
                0.5 * (self.A[2, 2] * x[2] * 2 + (self.A[2, 0] + self.A[0, 2]) * x[0] + (self.A[1, 0] + self.A[2, 1]) *
                       x[1]) + self.b[2]])
            x_new = x - self.alpha * grad

            self.trajectory.append(x_new.copy())  # Запись траектории

            if self.f(x_new) > self.f(x):
                self.alpha /= 2

            if np.linalg.norm(grad) < tol:
                break

            x = x_new

        return x, self.f(x)


# Параметры функции
A = np.array([[2, 3, 1],
              [2, 7, 2],
              [1, 3, 3]])
b = np.array([3, 4, 5])
x0 = np.array([1, 1, 5], dtype='float')
alpha = 0.01
h = 0.2

# Экземпляр класса
gradient = Gradient(A, b, x0, alpha, h)
x_min, f_min = gradient.gradient_descent()
print(x_min, f_min)
# Подготовка данных для визуализации
trajectory = np.array(gradient.trajectory)
x_vals, y_vals, z_vals = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]



# Сетка для отображения функции
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = 0.5 * (A[0, 0] * X**2 + A[1, 1] * Y**2 + A[0, 1] * X * Y) + b[0] * X + b[1] * Y

fig = plt.figure(figsize=(14, 10))

# 3D график
ax = fig.add_subplot(221, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.4, cmap='viridis')
ax.plot(x_vals, y_vals, z_vals, color='r', marker='o', label="Gradient Descent Trajectory")
ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color='g', label="Minimum", s=100)
ax.set_title("3D Gradient Descent Trajectory")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Function Value")
ax.legend()

# Проекция XY
ax2 = fig.add_subplot(222)
ax2.contour(X, Y, Z, levels=30, cmap='viridis')
ax2.plot(x_vals, y_vals, color='r', marker='o', label="Gradient Descent Trajectory")
ax2.scatter(x_vals[-1], y_vals[-1], color='g', label="Minimum", s=100)
ax2.set_title("XY Projection")
ax2.set_xlabel("X-axis")
ax2.set_ylabel("Y-axis")
ax2.legend()

# Проекция XZ
ax3 = fig.add_subplot(223)
ax3.contour(X, Z, Z, levels=30, cmap='viridis')
ax3.plot(x_vals, z_vals, color='r', marker='o', label="Gradient Descent Trajectory")
ax3.scatter(x_vals[-1], z_vals[-1], color='g', label="Minimum", s=100)
ax3.set_title("XZ Projection")
ax3.set_xlabel("X-axis")
ax3.set_ylabel("Z-axis")
ax3.legend()

# Проекция YZ
ax4 = fig.add_subplot(224)
ax4.contour(Y, Z, Z, levels=30, cmap='viridis')
ax4.plot(y_vals, z_vals, color='r', marker='o', label="Gradient Descent Trajectory")
ax4.scatter(y_vals[-1], z_vals[-1], color='g', label="Minimum", s=100)
ax4.set_title("YZ Projection")
ax4.set_xlabel("Y-axis")
ax4.set_ylabel("Z-axis")
ax4.legend()

plt.tight_layout()
plt.show()
