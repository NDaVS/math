import numpy as np
from matplotlib import pyplot as plt

class SolderingIron:
    """
    A class to represent a soldering iron and its temperature dynamics.

    Attributes:
    -----------
    P : float
        Power in watts.
    c : float
        Specific heat capacity in J/(g*K).
    m : float
        Mass in grams.
    T0 : float
        Initial temperature in Kelvins.
    Tenv : float
        Environmental temperature in Kelvins.
    Tmax : float
        Maximum temperature in Kelvins.
    Tmin : float
        Minimum temperature in Kelvins.
    alpha : float
        Heat transfer coefficient.
    S : float
        Surface area in square meters.
    sigma : float
        Stefan-Boltzmann constant.
    isOn : bool
        State of the soldering iron (on/off).
    """

    def __init__(self, P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma):
        """
        Constructs all the necessary attributes for the SolderingIron object.

        Parameters:
        -----------
        P : float
            Power in watts.
        c : float
            Specific heat capacity in J/(g*K).
        m : float
            Mass in grams.
        T0 : float
            Initial temperature in Kelvins.
        Tenv : float
            Environmental temperature in Kelvins.
        Tmax : float
            Maximum temperature in Kelvins.
        Tmin : float
            Minimum temperature in Kelvins.
        alpha : float
            Heat transfer coefficient.
        R : float
            Radius in meters.
        h : float
            Height in meters.
        sigma : float
            Stefan-Boltzmann constant.
        """
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

    def dTdt(self, T):
        """
        Calculates the rate of change of temperature without controller.

        Parameters:
        -----------
        T : float
            Current temperature in Kelvins.

        Returns:
        --------
        float
            Rate of change of temperature.
        """
        return (self.P - self.alpha * self.S * (T - self.Tenv) - self.sigma * self.S * (T ** 4 - self.Tenv ** 4)) / (self.m * self.c)

    def I(self, T):
        """
        Determines the state of the soldering iron (on/off) based on temperature.

        Parameters:
        -----------
        T : float
            Current temperature in Kelvins.

        Returns:
        --------
        int
            1 if the soldering iron is on, 0 if off.
        """
        if T > self.Tmax or (self.Tmin < T < self.Tmax and not self.isOn):
            self.isOn = False
            return 0
        self.isOn = True
        return 1

    def dTdt_with_controller(self, T):
        """
        Calculates the rate of change of temperature with controller.

        Parameters:
        -----------
        T : float
            Current temperature in Kelvins.

        Returns:
        --------
        float
            Rate of change of temperature.
        """
        return (self.P * self.I(T) - self.alpha * self.S * (T - self.Tenv) - self.sigma * self.S * (T ** 4 - self.Tenv ** 4)) / (self.m * self.c)

    def solve(self, t0, tn, n):
        """
        Solves the temperature dynamics without controller using the Runge-Kutta method.

        Parameters:
        -----------
        t0 : float
            Initial time.
        tn : float
            Final time.
        n : int
            Number of steps.

        Returns:
        --------
        tuple
            Time values and temperature values.
        """
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
        """
        Solves the temperature dynamics with controller using the Runge-Kutta method.

        Parameters:
        -----------
        t0 : float
            Initial time.
        tn : float
            Final time.
        n : int
            Number of steps.

        Returns:
        --------
        tuple
            Time values and temperature values.
        """
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

def plot_temperature_vs_time(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma):
    """
    Plots the temperature vs time without controller.

    Parameters:
    -----------
    P : float
        Power in watts.
    c : float
        Specific heat capacity in J/(g*K).
    m : float
        Mass in grams.
    T0 : float
        Initial temperature in Kelvins.
    Tenv : float
        Environmental temperature in Kelvins.
    Tmax : float
        Maximum temperature in Kelvins.
    Tmin : float
        Minimum temperature in Kelvins.
    alpha : float
        Heat transfer coefficient.
    R : float
        Radius in meters.
    h : float
        Height in meters.
    sigma : float
        Stefan-Boltzmann constant.
    """
    soldering_iron = SolderingIron(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    t_values, T_values = soldering_iron.solve(0, 1000, 1000)
    plt.subplot(1, 2, 1)
    plt.plot(t_values, T_values, label=fr"P={P}, c={c}, m={m}, $\alpha$={alpha}, S={soldering_iron.S:.4f}")
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time with Different Boundary Conditions')
    plt.legend()
    plt.grid(True)

def plot_temperature_vs_time_with_controller(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma):
    """
    Plots the temperature vs time with controller.

    Parameters:
    -----------
    P : float
        Power in watts.
    c : float
        Specific heat capacity in J/(g*K).
    m : float
        Mass in grams.
    T0 : float
        Initial temperature in Kelvins.
    Tenv : float
        Environmental temperature in Kelvins.
    Tmax : float
        Maximum temperature in Kelvins.
    Tmin : float
        Minimum temperature in Kelvins.
    alpha : float
        Heat transfer coefficient.
    R : float
        Radius in meters.
    h : float
        Height in meters.
    sigma : float
        Stefan-Boltzmann constant.
    """
    soldering_iron = SolderingIron(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    t_values, T_values = soldering_iron.solve_with_controller(0, 1000, 1000)
    plt.subplot(1, 2, 2)
    plt.plot(t_values, T_values, label=f"P={P}, c={c}, m={m}, $\\alpha$={alpha}, S={soldering_iron.S:.4f}")
    plt.axhline(y=Tmax, linestyle='--', label=f'Tmax={Tmax}')
    plt.axhline(y=Tmin, linestyle='--', label=f'Tmin={Tmin}')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time with Controller')
    plt.legend()
    plt.grid(True)

def main():
    """
    Main function to plot temperature vs time for different parameters.
    """
    C0 = 276
    P = 120
    c = 867
    m = 0.1
    T0 = 25.0 + C0
    Tenv = 25.0 + C0
    Tmax = 250 + C0
    Tmin = 200 + C0
    alpha = 2
    R = 0.005
    h = 0.1
    sigma = 5.67e-8

    plt.figure(figsize=(24, 8))
    plot_temperature_vs_time(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time(P * 2, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time(P, c / 2, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time(P, c, m * 2, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time(P, c, m, T0, Tenv, Tmax, Tmin, alpha * 2, R, h, sigma)
    plot_temperature_vs_time(P, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h * 2, sigma)
    plot_temperature_vs_time_with_controller(P * 2, c, m, T0, Tenv, Tmax, Tmin, alpha, R, h, sigma)
    plot_temperature_vs_time_with_controller(P, c / 2, m, T0, Tenv, 200 + C0, 150 + C0, alpha, R, h, sigma)
    plot_temperature_vs_time_with_controller(P, c, m, T0, Tenv, 150 + C0, 130 + C0, alpha, R, h, sigma)
    plt.show()

if __name__ == '__main__':
    main()