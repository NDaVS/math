import numpy as np
from matplotlib import pyplot as plt


class SolderingIron:
    def __init__(self, P, c, m, T0, Tenv, Tmax, Tmin, k, R, h, sigma):
        self.P = P
        self.c = c
        self.m = m
        self.T0 = T0
        self.Tenv = Tenv
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.k = k
        self.S = 2 * np.pi * R * h
        self.sigma = sigma
        self.isOn = True

    def dTdt(self, T):
        return (self.P - self.k * self.S * (T - self.Tenv) - self.sigma * self.S * (T ** 4 - self.Tenv ** 4)) / (
                self.m * self.c)

    def I(self, T):
        if T > self.Tmax:
            self.isOn = False
        if T < self.Tmin:
            self.isOn = True

    def dTdt_with_controller(self, T):
        self.I(T)
        return (self.P * self.isOn - self.k * self.S * (T - self.Tenv) - self.sigma * self.S * (
                T ** 4 - self.Tenv ** 4)) / (self.m * self.c)

    def solve(self, t0, tn, n):
        h = (tn - t0) / n
        T = self.T0
        t_values, T_values = [t0], [self.T0]

        for _ in range(n):
            k1 = h * self.dTdt(T)
            k2 = h * self.dTdt(T + 0.5 * k1)
            k3 = h * self.dTdt(T + 0.5 * k2)
            k4 = h * self.dTdt(T + k3)
            T += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t_values.append(t_values[-1] + h)
            T_values.append(T)

        return t_values, T_values

    def solve_with_controller(self, t0, tn, n):
        h = (tn - t0) / n
        T = self.T0
        t_values, T_values = [t0], [self.T0]

        for _ in range(n):
            k1 = h * self.dTdt_with_controller(T)
            k2 = h * self.dTdt_with_controller(T + 0.5 * k1)
            k3 = h * self.dTdt_with_controller(T + 0.5 * k2)
            k4 = h * self.dTdt_with_controller(T + k3)
            T += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t_values.append(t_values[-1] + h)
            T_values.append(T)

        return t_values, T_values



