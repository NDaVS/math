import numpy as np
import matplotlib.pyplot as plt

g = 9.81
l = 1.0
mu = 0.0
m = 1.0
F = 0
omega_f = np.sqrt(g/l)
theta0 = np.pi / 10
omega0 = 0

def pendulum_eq(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / l) * theta - (mu / m) * omega + F * np.sin(omega_f * t)
    return np.array([dtheta_dt, domega_dt])

def pendulum_eq_lin(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta) - (mu / m) * omega + F * np.sin(omega_f * t)
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

t_span = (0, 25)
dt = 0.025

t_vals, sol  = runge_kutta4(pendulum_eq, [theta0, omega0], t_span, dt)
t_vals_lin, sol_lin = runge_kutta4(pendulum_eq, [theta0, omega0], t_span, dt)
