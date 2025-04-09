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

    def plot_trajectories(self, initial_conditions, t0, tn, h):
        plt.figure(figsize=(8, 8))
        for idx, y0 in enumerate(initial_conditions):
            _, y_values = self.runge_kutta(y0, t0, tn, h)
            x, y = y_values[:, 0], y_values[:, 1]
            plt.plot(x, y, label=f"Траектория {idx + 1}")
            plt.scatter([y0[0]], [y0[1]], label=f"Нач. точка {y0[0], y0[1]}")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Движение во вращающейся системе коодинат")
        plt.legend()
        plt.grid()
        plt.axis("equal")
        plt.show()


# Example usage
if __name__ == "__main__":
    omega = 10.0  # Angular velocity
    phi = np.pi / 2.5  # Latitude (45 degrees)
    print(np.degrees(phi))
    initial_conditions = [
        [1.0, 0.0, 0.0, 1.0],  # Initial state [x, y, vx, vy]
        [0.5, 0.5, -0.5, 0.5],
        [-1.0, 0.0, 0.0, -1.0],
        [5, 3, -3, 3],
        [5, 3, -4, -2],
        [5, 3, 1, -3]
    ]
    t0, tn = 0, 100  # Time range
    h = 0.01  # Time step

    motion = CircleMotion(omega, phi)
    motion.plot_trajectories(initial_conditions, t0, tn, h)