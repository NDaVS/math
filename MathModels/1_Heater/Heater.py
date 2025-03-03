import numpy as np
from matplotlib import pyplot as plt


class SolderingIron:
    def __init__(self, P: float, c: float, m: float, T0: float, Tenv: float, Tmax: float, Tmin: float, alpha: float,
                 R: float, h: float,
                 sigma: float):
        self.P = P
        self.c = c
        self.m = m
        self.T0 = T0
        self.Tenv = Tenv
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.alpha = alpha
        self.S = 2 * np.pi * R * h
        self.sigma = sigma
        self.isOn = True

    def dTdt(self, T: float) -> float:
        return (self.P - self.alpha * self.S * (T - self.Tenv) - self.sigma * self.S * (T ** 4 - self.Tenv ** 4)) / (
                self.m * self.c)

    def I(self, T: float) -> int:
        if T > self.Tmax or (self.Tmin < T < self.Tmax and not self.isOn):
            self.isOn = False
            return 0
        self.isOn = True
        return 1

    def dTdt_with_controller(self, T: float) -> float:
        return (self.P * self.I(T) - self.alpha * self.S * (T - self.Tenv) - self.sigma * self.S * (
                    T ** 4 - self.Tenv ** 4)) / (
                self.m * self.c)

    def solve(self, t0: float, tn: float, n: int):
        h = (tn - t0) / n
        T = self.T0
        t = t0
        T_values = [self.T0]
        t_values = [t0]

        for _ in range(n):
            k1 = h * self.dTdt(T)
            k2 = h * self.dTdt(T + 0.5 * k1)
            k3 = h * self.dTdt(T + 0.5 * k2)
            k4 = h * self.dTdt(T + k3)
            T += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += h
            T_values.append(T)
            t_values.append(t)

        return t_values, T_values

    def solve_with_controller(self, t0: float, tn: float, n: int):
        h = (tn - t0) / n
        T = self.T0
        t = t0
        T_values = [self.T0]
        t_values = [t0]

        for _ in range(n):
            k1 = h * self.dTdt_with_controller(T)
            k2 = h * self.dTdt_with_controller(T + 0.5 * k1)
            k3 = h * self.dTdt_with_controller(T + 0.5 * k2)
            k4 = h * self.dTdt_with_controller(T + k3)
            T += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += h
            T_values.append(T)
            t_values.append(t)

        return t_values, T_values


def plot_temperature_vs_time(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma):
    soldering_iron = SolderingIron(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    t_values, T_values = soldering_iron.solve(0, 1000, 1000)
    plt.subplot(1, 2, 1)  # Right plot
    plt.plot(t_values, T_values, label=fr"P={P}, c={c}, m={m}, $\alpha$ ={alpha}, S={soldering_iron.S:.4f}")
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time with Different Boundary Conditions')
    plt.legend()
    plt.grid(True)

def plot_temperature_vs_time_with_controller(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma):
    soldering_iron = SolderingIron(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    t_values, T_values = soldering_iron.solve_with_controller(0, 1000, 1000)
    plt.subplot(1, 2, 2)  # Right plot
    plt.plot(t_values, T_values, label=fr"P={P}, c={c}, m={m}, $\alpha$ ={alpha}, S={soldering_iron.S:.4f}")
    plt.axhline(y=Tmax, linestyle='--', label=f'Tmax ={Tmax}')
    plt.axhline(y=Tmin, linestyle='--', label=f'Tmin={Tmin}')



def main():
    C0 = 276
    P = 120  # Power in watts
    c = 867  # Specific heat capacity in J/(g*K)
    m = 0.1  # Mass in grams
    T0 = 25.0 + C0  # Initial temperature in Celsius
    Tenv = 25.0 + C0  # Environmental temperature in Celsius
    Tmax = 250 + C0
    Tmin = 200 + C0
    alpha = 2  # Heat transfer coefficient
    R = 0.005  # Radius in meters
    h = 0.1  # Height in meters
    sigma = 5.67e-8  # Stefan-Boltzmann constant

    plt.figure(figsize=(24, 8))

    # Original parameters
    plot_temperature_vs_time(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time(P * 2, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time(P, c / 2, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time(P, c, m * 2, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time(P, c, m, T0, Tenv, Tmax, Tmin, alpha * 2, R, h, sigma)
    plot_temperature_vs_time(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h * 2, sigma)
    # Plot with different boundary conditions
    plot_temperature_vs_time_with_different_boundary_conditions(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plt.show()


if __name__ == '__main__':
    main()


def plot_temperature_vs_time_with_different_boundary_conditions(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma):
    plt.subplot(1, 2, 1)  # Left plot
    plot_temperature_vs_time_with_controller(P * 2, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time_with_controller(P, c / 2, m, T0, Tenv, 200 + 276, 150 + 276, alpha, R, h, sigma)
    plot_temperature_vs_time_with_controller(P, c, m, T0, Tenv, 150 + 276, 130 + 276, alpha, R, h, sigma)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time with Different Boundary Conditions')
    plt.legend()
    plt.grid(True)
