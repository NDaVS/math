import numpy as np
import matplotlib.pyplot as plt

g = 9.81
l = 1.0
b = 0.0
m = 1.0
F = 0
omega_f = np.sqrt(g/l)
theta0 = np.pi / 10
omega0 = 0

def pendulum_eq_lin(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / l) * theta - (b / m) * omega + F * np.sin(omega_f * t)
    return np.array([dtheta_dt, domega_dt])

def pendulum_eq(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta) - (b / m) * omega + F * np.sin(omega_f * t)
    return np.array([dtheta_dt, domega_dt])


def runge_kutta4(f, y0, t_span, dt):
    t_values = np.arange(t_span[0], t_span[1] + dt, dt)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]

        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 2, y + k1 / 2)
        k3 = dt * f(t + dt / 2, y + k2 / 2)
        k4 = dt * f(t + dt, y + k3)

        y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values.T

t_span = (0, 15)
dt = 0.025

t_vals, sol  = runge_kutta4(pendulum_eq, [theta0, omega0], t_span, dt)
t_vals_lin, sol_lin = runge_kutta4(pendulum_eq_lin, [theta0, omega0], t_span, dt)


def draw(t_vals1, theta_vals1, omega_vals1, t_vals2, theta_vals2, omega_vals2):
    plt.figure(figsize=(12, 6))

    # График угла
    plt.subplot(1, 2, 1)
    plt.plot(t_vals1, theta_vals1, 'b', label=r'$\theta(t)$ (Модель нелинейная)')
    plt.plot(t_vals2, theta_vals2, 'g', label=r'$\theta(t)$ (Модель линейная)')
    plt.plot(t_vals1[0], theta_vals1[0], 'bo',markersize=10, label='Начальное положение (Модель нелинейная)')
    plt.plot(t_vals2[0], theta_vals2[0], 'go', label='Начальное положение (Модель линейная)')
    plt.xlabel('Время, с')
    plt.ylabel('Угол (рад)')
    plt.title('Динамика математического маятника')
    plt.legend(loc='upper right')
    plt.grid()

    # График фазового портрета
    plt.subplot(1, 2, 2)
    plt.plot(theta_vals1, omega_vals1, 'b', label=r'Фазовый портрет (Модель нелинейная)')
    plt.plot(theta_vals2, omega_vals2, 'g', label=r'Фазовый портрет (Модель линейная)')
    plt.plot(theta_vals1[0], omega_vals1[0], 'bo',markersize=10, label='Начальное положение (Модель нелинейная)')
    plt.plot(theta_vals2[0], omega_vals2[0], 'go', label='Начальное положение (Модель линейная)')
    plt.title('Фазовый портрет математического маятника')
    plt.xlabel('Угол (рад)')
    plt.ylabel('Угловая скорость (рад/с)')
    plt.legend(loc='upper right')
    plt.grid()

    plt.tight_layout()
    plt.show()

# Вызов функции для отрисовки
draw(t_vals, sol[0], sol[1], t_vals_lin, sol_lin[0], sol_lin[1])
