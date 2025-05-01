import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.integrate import quad
from scipy.linalg import solve

# Параметры
a, b = -1.0, 1.0
degree = 3
n_internal = 6  # число внутренних контрольных точек
n = n_internal + 2  # включая граничные
k = degree

# Узловой вектор с кратными концами
t = np.concatenate((
    np.full(k, a),
    np.linspace(a, b, n - k + 1),
    np.full(k, b)
))

# Индексы базисов с обнулением на концах
indices = range(1, n - 1)
basis_count = len(indices)

# Сетка для интегрирования
x_vals = np.linspace(a, b, 500)

# Построение B-сплайна вручную
def bspline(i, t, k, x):
    return BSpline.basis_element(t[i:i+k+2], extrapolate=False)(x)

def bspline_deriv(i, t, k, x):
    return BSpline.basis_element(t[i:i+k+2], extrapolate=False).derivative()(x)

# Инициализация матриц
A = np.zeros((basis_count, basis_count))
M = np.zeros((basis_count, basis_count))
F = np.zeros(basis_count)

# Сборка матриц
for i_idx, i in enumerate(indices):
    for j_idx, j in enumerate(indices):
        A[i_idx, j_idx], _ = quad(
            lambda x: bspline_deriv(i, t, k, x) * bspline_deriv(j, t, k, x),
            a, b, epsabs=1e-6, epsrel=1e-6
        )
        M[i_idx, j_idx], _ = quad(
            lambda x: (1 + x**2) * bspline(i, t, k, x) * bspline(j, t, k, x),
            a, b
        )
    F[i_idx], _ = quad(lambda x: bspline(i, t, k, x), a, b)

# Левая часть системы
K = A + M

# Решение СЛАУ
c = solve(K, -F)

# Построение решения
u_approx = np.zeros_like(x_vals)
for i_idx, i in enumerate(indices):
    u_approx += c[i_idx] * bspline(i, t, k, x_vals)

plt.plot(x_vals, u_approx, label='Численное решение (B-сплайны)')
plt.title('Решение краевой задачи через B-сплайны')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid()
plt.legend()
plt.show()
