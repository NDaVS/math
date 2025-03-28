import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Параметры модели
alpha = 2 # коэффициент размножения жертв
beta = 1  # вероятность поедания жертв
delta = 0.1  # коэффициент прироста хищников
gamma = 0.8  # смертность хищников


def lotka_volterra(x, y):
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])


def runge_kutta_4(f, x0, y0, T, dt):
    t_values = np.arange(0, T, dt)
    x_values = np.zeros(len(t_values))
    y_values = np.zeros(len(t_values))

    x_values[0] = x0
    y_values[0] = y0

    for i in range(1, len(t_values)):
        x, y = x_values[i - 1], y_values[i - 1]
        k1 = dt * f(x, y)
        k2 = dt * f(x + 0.5 * k1[0], y + 0.5 * k1[1])
        k3 = dt * f(x + 0.5 * k2[0], y + 0.5 * k2[1])
        k4 = dt * f(x + k3[0], y + k3[1])

        x_values[i] = x + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        y_values[i] = y + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6

    return t_values, x_values, y_values


# Начальные условия и параметры
initial_conditions = [(6, 1), (5,1.3), (2,1), (8,2)]
T = 11  # Время моделирования
n = 2000 # Количество шагов
dt = T / n  # Шаг интегрирования

sns.set_theme(style="darkgrid")
plt.figure(figsize=(12, 6))

for x0, y0 in initial_conditions:
    t_values, x_values, y_values = runge_kutta_4(lotka_volterra, x0, y0, T, dt)
    plt.plot(t_values, x_values, label=f'Жертвы (x0={x0}, y0={y0})', linewidth=2)
    plt.plot(t_values, y_values, label=f'Хищники (x0={x0}, y0={y0})', linewidth=2)

plt.xlabel("Время")
plt.ylabel("Популяция")
plt.title("Динамика популяций в модели Лотки-Вольтерра (Метод Рунге-Кутта)")
plt.legend()
plt.show()

# Фазовые портреты
plt.figure(figsize=(8, 6))

for x0, y0 in initial_conditions:
    t_values, x_values, y_values = runge_kutta_4(lotka_volterra, x0, y0, T, dt)
    plt.plot(x_values, y_values, linewidth=2, label=f'Нач. усл. ({x0}, {y0})')
    plt.plot(x_values[0], y_values[0], marker='o', markersize=5)

plt.xlabel("Жертвы")
plt.ylabel("Хищники")
plt.title("Фазовые портреты системы Лотки-Вольтерра (Метод Рунге-Кутта)")
plt.legend()
plt.show()
