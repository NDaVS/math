import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline
from tabulate import tabulate

# Параметры задачи
N = 20
a, b = 0.0, 1.0
x = np.linspace(a, b, N + 1)
h = (b - a) / N

# Коэффициенты из уравнения
def p(x): return 4 * x / (x**2 + 1)
def q(x): return -1 / (x**2 + 1)
def f(x): return -3 / (x**2 + 1)**2
def u_exact(x): return 1 / (x**2 + 1)

# Построение матрицы и правой части
A = np.zeros((N + 1, N + 1))
B = np.zeros(N + 1)

# Внутренние точки
for i in range(1, N):
    xi = x[i]
    pi = p(xi)
    qi = q(xi)
    fi = f(xi)

    A[i, i - 1] = 1 / h**2 - pi / (2 * h)
    A[i, i] = -2 / h**2 + qi
    A[i, i + 1] = 1 / h**2 + pi / (2 * h)
    B[i] = fi

# Краевые условия
# u'(0) = 0 → (u[1] - u[-1]) / (2h) = 0 → u[1] = u[1], u[0] = u[1]
A[0, 0] = -1
A[0, 1] = 1
B[0] = 0

# u(1) = 0.5
A[N, N] = 1
B[N] = 0.5

# Решение СЛАУ
u_numeric = np.linalg.solve(A, B)

# Построение B-сплайна по численному решению
k = 3  # степень сплайна
t_internal = np.linspace(a, b, N - k + 1)[1:-1]
t = np.concatenate(([a] * (k + 1), t_internal, [b] * (k + 1)))
spl = make_lsq_spline(x, u_numeric, t, k=k)

# Тестовая сетка
x_fine = np.linspace(a, b, 500)
u_interp = spl(x_fine)
u_true = u_exact(x_fine)
error = np.abs(u_interp - u_true)

# Таблица ошибок
u_grid_true = u_exact(x)
error_grid = np.abs(u_numeric - u_grid_true)

table = []
for xi, ui, ue, err in zip(x, u_numeric, u_grid_true, error_grid):
    table.append([f"{xi:.3f}", f"{ui:.6f}", f"{ue:.6f}", f"{err:.2e}"])

print("\nТаблица значений (B-сплайн):")
print(tabulate(table, headers=["x", "Численное u", "Точное u", "Ошибка"], tablefmt="grid"))

# График
plt.figure(figsize=(10, 6))
plt.plot(x_fine, u_interp, 'r-', label='B-сплайн')
plt.plot(x_fine, u_true, 'b--', label='Точное решение')
plt.plot(x, u_numeric, 'ko', label='Узлы')
plt.title("Метод конечных разностей + B-сплайн")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nМаксимальная ошибка: {np.max(error):.6e}")
