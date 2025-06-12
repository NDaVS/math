import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os


class FluidSimulation:
    def __init__(self, n=100, t_max=10.0, dt=0.01, save_frames=True):
        self.n = n
        self.t_max = t_max
        self.dt = dt
        self.save_frames = save_frames

        self.output_dir = 'lab6_images_fd'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_velocity_field(self, velocity_field, t, domain, n=20, title="Поле скоростей", filename=None):
        x_min, x_max, y_min, y_max = domain
        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)
        xx, yy = np.meshgrid(x, y)

        U = np.zeros_like(xx)
        V = np.zeros_like(yy)

        for i in range(n):
            for j in range(n):
                U[j, i], V[j, i] = velocity_field(xx[j, i], yy[j, i], t)

        speed = np.sqrt(U ** 2 + V ** 2)

        fig, ax = plt.subplots(figsize=(10, 8))
        strm = ax.streamplot(xx, yy, U, V, color=speed / 2, cmap=cm.viridis, linewidth=1.5, density=1, arrowsize=2)
        fig.colorbar(strm.lines, ax=ax, label='Скорость')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.grid(True)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def solve_finite_difference(self, velocity_field, initial_condition, domain, experiment_name, plot_every=10):
        x_min, x_max, y_min, y_max = domain
        n = self.n

        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)
        xx, yy = np.meshgrid(x, y)

        dx = (x_max - x_min) / (n - 1)
        dy = (y_max - y_min) / (n - 1)

        u = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                u[j, i] = initial_condition(x[i], y[j])

        n_steps = int(self.t_max / self.dt)

        for step in range(n_steps):
            t = step * self.dt

            u_new = u.copy()

            vx = np.zeros((n, n))
            vy = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    vx[j, i], vy[j, i] = velocity_field(x[i], y[j], t)

            # Проверка условия CFL
            cfl_x = np.max(np.abs(vx)) * self.dt / dx
            cfl_y = np.max(np.abs(vy)) * self.dt / dy
            if cfl_x > 1 or cfl_y > 1:
                print(f"Нарушено условие CFL! cfl_x = {cfl_x:.3f}, cfl_y = {cfl_y:.3f}")

            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    if vx[j, i] >= 0:
                        du_dx = (u[j, i] - u[j, i - 1]) / dx
                    else:
                        du_dx = (u[j, i + 1] - u[j, i]) / dx

                    if vy[j, i] >= 0:
                        du_dy = (u[j, i] - u[j - 1, i]) / dy
                    else:
                        du_dy = (u[j + 1, i] - u[j, i]) / dy

                    u_new[j, i] = u[j, i] - self.dt * (vx[j, i] * du_dx + vy[j, i] * du_dy)

            # Граничные условия (периодические)
            u_new[0, :] = u_new[1, :]  # Нижняя граница
            u_new[-1, :] = u_new[-2, :]  # Верхняя граница
            u_new[:, 0] = u_new[:, 1]  # Левая граница
            u_new[:, -1] = u_new[:, -2]  # Правая граница

            u = u_new

            if step % plot_every == 0 or step == n_steps - 1:
                print(f"Шаг {step + 1}/{n_steps}, Время: {t:.2f}")

                fig, ax = plt.subplots(figsize=(12, 10))

                contour = ax.contourf(xx, yy, u, cmap='viridis', levels=50)
                cbar = plt.colorbar(contour, ax=ax, label='Значение поля u')

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f'{experiment_name}: t = {t:.2f}')

                if self.save_frames:
                    plt.savefig(f'{self.output_dir}/{experiment_name}_t{t:.2f}.png', dpi=300,
                                bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()

        return u


class VelocityField:
    @staticmethod
    def spiral_flow(x, y, t):
        r = np.sqrt((x - 5) ** 2 + (y - 5) ** 2)  # Расстояние от центра (5, 5)
        theta = np.arctan2(y - 5, x - 5) + 0.1 * t  # Угол с учетом времени
        vx = -np.sin(theta) / r  # Горизонтальная скорость
        vy = np.cos(theta) / r   # Вертикальная скорость
        return vx, vy

    @staticmethod
    def wave_flow(x, y, t):
        vx = 0.5 * np.sin(0.5 * (x + t))  # Горизонтальная скорость с волновым эффектом
        vy = 0.5 * np.cos(0.5 * (y + t))  # Вертикальная скорость с волновым эффектом
        return vx, vy

    @staticmethod
    def source_sink_flow(x, y, t):
        vx = - np.pi * np.sin(2 * np.pi * x/10) * np.cos(np.pi * y/10)
        vy = 2* np.pi * np.sin( np.pi * y/10) * np.cos(2 * np.pi * x/10)

        return vx, vy


class InitialCondition:
    @staticmethod
    def diagonal_line(x,y):
        if np.abs(x-y) < 1:
            return 1

        return 0

    @staticmethod
    def moved_circle(x, y):
        if ((x - 2.5) ** 2 + (y - 5) ** 2) < 4:
            return 1.0

        return 0.0

    @staticmethod
    def circle(x, y):
        if (x - 5) ** 2 + (y - 5) ** 2 < 4:
            return 1.0

        return 0.0

    @staticmethod
    def vertical_line(x,y):
        if np.abs(x-5) < 1:
            return 1

        return 0

    @staticmethod
    def horizontal_line(x, y):
        if np.abs(y-5) < 1:
            return 1

        return 0


class Experiment:
    def __init__(self, simulation, name, velocity_field, initial_condition, domain=(0, 10, 0, 10)):
        self.simulation = simulation
        self.name = name
        self.velocity_field = velocity_field
        self.initial_condition = initial_condition
        self.domain = domain

    def run(self):
        print(f"\n=== {self.name} (метод конечных разностей) ===")

        # Визуализация поля скоростей
        self.simulation.plot_velocity_field(
            self.velocity_field, 0, self.domain,
            title=f"{self.name} (метод конечных разностей)",
            filename=f"{self.simulation.output_dir}/{self.name.lower().replace(' ', '_')}_fd_velocity_field.png"
        )

        # Решение методом конечных разностей
        u_final = self.simulation.solve_finite_difference(
            self.velocity_field, self.initial_condition, self.domain,
            self.name.lower().replace(' ', '_'),
            plot_every=100
        )

        return u_final


class ExperimentManager:
    def __init__(self):
        self.simulation = FluidSimulation()
        self.experiments = []
        self.results = {}

    def add_experiment(self, name, velocity_field, initial_condition, domain=(0, 10, 0, 10)):
        experiment = Experiment(self.simulation, name, velocity_field, initial_condition, domain)
        self.experiments.append(experiment)

    def run_all(self):
        print("Метод конечных разностей")

        for experiment in self.experiments:
            u_final = experiment.run()
            self.results[experiment.name] = u_final

        return self.results

    def compare_methods(self):
        print("\n=== Сравнение методов: Лагранжевы частицы vs Конечные разности ===")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].set_title("Метод Лагранжевых частиц")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].grid(True)

        axes[1].set_title("Метод конечных разностей")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.simulation.output_dir}/comparison_methods.png", dpi=300, bbox_inches='tight')
        plt.close()


import time

def main():
    manager = ExperimentManager()

    # Добавление экспериментов
    experiments = [
        # ("Эксперимент 1: Вихревое течение", VelocityField.spiral_flow, InitialCondition.horizontal_line),
        ("Эксперимент 2: Сдвиговое течение", VelocityField.wave_flow, InitialCondition.diagonal_line),
        # ("Эксперимент 3: сложные системы", VelocityField.source_sink_flow, InitialCondition.circle),
    ]

    for name, velocity_field, initial_condition in experiments:
        start_time = time.time()
        manager.add_experiment(name, velocity_field, initial_condition)
        end_time = time.time()
        print(f"Время добавления {name}: {end_time - start_time:.2f} секунд")

    # Запуск всех экспериментов
    start_time = time.time()
    results = manager.run_all()
    end_time = time.time()
    print(f"Время выполнения всех экспериментов: {end_time - start_time:.2f} секунд")

    # # Сравнение методов
    # start_time = time.time()
    # manager.compare_methods()
    # end_time = time.time()
    # print(f"Время сравнения методов: {end_time - start_time:.2f} секунд")

if __name__ == "__main__":
    main()
