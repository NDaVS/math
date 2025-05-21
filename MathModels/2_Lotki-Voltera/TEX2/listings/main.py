import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


alpha = 2 
beta = 2  
delta = 1 
gamma = 4 


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



initial_conditions = [(1, 3), (6, 1), (4,1)]
T = 10  
n = 1000
dt = T / n 

