import numpy as np
import matplotlib.pyplot as plt


class CircleMotion:
    def __init__(self, omega, phi):
        self.omega = omega
        self.phi = phi

    def f(self, t, x):
        return np.array([
            x[2],
            x[3],
            2 * self.omega * x[3] * np.cos(self.phi),
            -2 * self.omega * x[2] * np.cos(self.phi),
        ])

    def runge_kutta(self, y0, t0, tn, h):
        num = int(np.ceil((tn - t0) / h))
        t_values = np.linspace(t0, tn, num=num)
        y_values = np.zeros((num, len(y0)))
        y_values[0] = y0

        for i in range(num - 1):
            k1 = h * self.f(t_values[i], y_values[i])
            k2 = h * self.f(t_values[i] + h / 2, y_values[i] + k1 / 2)
            k3 = h * self.f(t_values[i] + h / 2, y_values[i] + k2 / 2)
            k4 = h * self.f(t_values[i] + h, y_values[i] + k3)
            y_values[i + 1] = y_values[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t_values, y_values
