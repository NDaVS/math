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


def plot_temperature_vs_time(P, c, m, T0, Tenv, k, R, h, sigma):
    soldering_iron = SolderingIron(P, c, m, T0, Tenv, 0, 0, k, R, h, sigma)
    t_values, T_values = soldering_iron.solve(0, 3600, 1000)
    plt.plot(t_values, T_values, label=fr"P={P}, c={c}, m={m}, k={k}, S={soldering_iron.S:.4f}")
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time with Different Boundary Conditions')
    plt.legend()
    plt.grid(True)


def make_plot_for_controller(soldering_iron1, soldering_iron2, ):
    t_values, T_values = soldering_iron1.solve_with_controller(0, 3600, 1000)
    plt.plot(t_values, T_values)
    plt.axhline(y=soldering_iron1.Tmax, linestyle='--', color='red', label=f'Tmax_1={soldering_iron1.Tmax}')
    plt.axhline(y=soldering_iron1.Tmin, linestyle='--', color='blue', label=f'Tmin_1={soldering_iron1.Tmin}')

    t_values, T_values = soldering_iron2.solve_with_controller(0, 3600, 1000)
    plt.plot(t_values, T_values)
    plt.axhline(y=soldering_iron2.Tmax, linestyle='--', color='green', label=f'Tmax_2={soldering_iron1.Tmax}')
    plt.axhline(y=soldering_iron2.Tmin, linestyle='--', color='purple', label=f'Tmin_2={soldering_iron1.Tmin}')


def plot_temperature_vs_time_with_controller(C0, P, c, m, T0, Tenv, k, R, h, sigma):
    soldering_iron_1 = SolderingIron(P, c, m, T0, Tenv, 250 + C0, 200 + C0, k, R, h, sigma)
    soldering_iron_2 = SolderingIron(P, c, m, T0, Tenv, 190 + C0, 180 + C0, k, R, h, sigma)

    make_plot_for_controller(soldering_iron_1, soldering_iron_2)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time with Controller')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    C0 = 276
    P = 35
    c = 375
    m = 0.25
    T0 = 25.0 + C0
    Tenv = 25.0 + C0
    k = 2
    R = 0.003
    h = 0.05
    sigma = 5.67e-8

    plt.figure(figsize=(12, 8))
    plot_temperature_vs_time(P, c, m, T0, Tenv, k, R, h, sigma)
    plot_temperature_vs_time(P * 2, c, m, T0, Tenv, k, R, h, sigma)
    plot_temperature_vs_time(P, c / 2, m, T0, Tenv, k, R, h, sigma)
    plot_temperature_vs_time(P, c, m * 2, T0, Tenv, k, R, h, sigma)
    plot_temperature_vs_time(P, c, m, T0, Tenv, k * 2, R, h, sigma)
    plot_temperature_vs_time(P, c, m, T0, Tenv, k, R, h * 2, sigma)
    plt.axhline(y=895.53, linestyle='--')

    plt.show()
    plot_temperature_vs_time_with_controller(C0, P * 2, c, m, T0, Tenv, k, R, h, sigma)
    plot_temperature_vs_time_with_controller(C0, P, c / 2, m, T0, Tenv, k, R, h, sigma)
    plot_temperature_vs_time_with_controller(C0, P, c, m, T0, Tenv, k, R, h, sigma)


if __name__ == '__main__':
    main()
