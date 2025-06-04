import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata


class VelocityField:
    @staticmethod
    def vortex_flow(x, y, t):
        vx = -0.1 * (y - 5)
        vy = 0.1 * (x - 5)
        return vx, vy

    @staticmethod
    def shear_flow(x, y, t):
        vx = 0.5 * (1 + np.sin(y))
        vy = 0.1 * np.cos(x)
        return vx, vy

    @staticmethod
    def source_sink_flow(x, y, t):
        vx = - np.pi * np.sin(2 * np.pi * x/10) * np.cos(np.pi * y/10)
        vy = 2* np.pi * np.sin( np.pi * y/10) * np.cos(2 * np.pi * x/10)

        return vx, vy


class InitialCondition:
    @staticmethod
    def gaussian(x, y, x0, y0, sigma):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    @staticmethod
    def circle(x, y, x0, y0, r):
        return np.where((x - x0)**2 + (y - y0)**2 < r**2, 1.0, 0.0)


class ParticleGenerator:

    @staticmethod
    def cross_pattern(n_particles, x_min=2, x_max=8, y_min=2, y_max=8):
        n_particles_line = int(np.sqrt(n_particles))
        x_line = np.linspace(x_min, x_max, n_particles_line)
        y_line = np.linspace(y_min, y_max, n_particles_line)

        x_cross = np.concatenate([x_line, np.ones(n_particles_line) * 5])
        y_cross = np.concatenate([np.ones(n_particles_line) * 5, y_line])

        return np.column_stack((x_cross, y_cross))

    @staticmethod
    def grid_pattern(n_particles, x_min=2, x_max=8, y_min=2, y_max=8):
        n_side = int(np.sqrt(n_particles))
        x_grid = np.linspace(x_min, x_max, n_side)
        y_grid = np.linspace(y_min, y_max, n_side)
        X, Y = np.meshgrid(x_grid, y_grid)

        return np.column_stack((X.flatten(), Y.flatten()))

    @staticmethod
    def radial_pattern(n_particles, x0=5., y0=5., r_max=3.):
        np.random.seed(42)
        r = r_max * np.sqrt(np.random.random(n_particles))
        theta = 2 * np.pi * np.random.random(n_particles)

        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)

        return np.column_stack((x, y)), r / r_max


class Integrator:
    @staticmethod
    def rk4_step(x, y, t, dt, velocity_field):
        k1_x, k1_y = velocity_field(x, y, t)

        k2_x, k2_y = velocity_field(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, t + 0.5 * dt)

        k3_x, k3_y = velocity_field(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, t + 0.5 * dt)

        k4_x, k4_y = velocity_field(x + dt * k3_x, y + dt * k3_y, t + dt)

        x_new = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_new = y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

        return x_new, y_new


class Visualizer:
    def __init__(self, output_dir='lab6_images'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_velocity_field(self, velocity_field, t, domain, nx=20, ny=20, title="Поле скоростей", filename=None):
        x_min, x_max, y_min, y_max = domain
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(nx):
            for j in range(ny):
                U[j, i], V[j, i] = velocity_field(X[j, i], Y[j, i], t)

        speed = np.sqrt(U**2 + V**2)

        fig, ax = plt.subplots(figsize=(10, 8))
        strm = ax.streamplot(X, Y, U, V, color=speed, cmap=cm.viridis, linewidth=1, density=1.5, arrowsize=1.5)
        fig.colorbar(strm.lines, ax=ax, label='Скорость')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.grid(True)

        if filename:
            full_path = f"{self.output_dir}/{filename}"
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_particles(self, x, y, u, t, domain, experiment_name, show_trajectories=False, trajectories=None, save=False):
        x_min, x_max, y_min, y_max = domain

        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

        points = np.column_stack((x, y))
        grid_u = griddata(points, u, (grid_x, grid_y), method='cubic', fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 10))

        contour = ax.contourf(grid_x, grid_y, grid_u, cmap='viridis', levels=50, alpha=0.7)
        cbar = plt.colorbar(contour, ax=ax, label='Значение поля u')

        scatter = ax.scatter(x, y, c=u, cmap='plasma', s=30, edgecolor='k', linewidth=0.5, alpha=0.8)

        if show_trajectories and trajectories is not None:
            n_particles = len(x)
            n_trajectories = min(50, n_particles)  # Ограничиваем количество отображаемых траекторий
            current_step = trajectories['t'].shape[0] - 1

            for i in range(0, n_particles, n_particles // n_trajectories):
                ax.plot(trajectories['x'][:current_step + 1, i], trajectories['y'][:current_step + 1, i],
                        'k-', linewidth=0.5, alpha=0.3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{experiment_name}: t = {t:.2f}')
        ax.grid(True)

        if save:
            full_path = f"{self.output_dir}/{experiment_name}_t{t:.2f}.png"
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_trajectories(self, history, domain, experiment_name, save=False):
        x_min, x_max, y_min, y_max = domain
        n_particles = history['x'].shape[1]

        fig, ax = plt.subplots(figsize=(12, 10))

        for i in range(n_particles):
            ax.plot(history['x'][:, i], history['y'][:, i], '-', linewidth=0.8, alpha=0.5,
                    color=plt.cm.plasma(history['u'][0, i] / np.max(history['u'][0])))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{experiment_name}: Траектории частиц')
        ax.grid(True)

        if save:
            full_path = f"{self.output_dir}/{experiment_name}_trajectories.png"
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class LagrangianSimulation:
    def __init__(self, n_particles=1000, t_max=10.0, dt=0.05, save_frames=True):
        self.n_particles = n_particles
        self.t_max = t_max
        self.dt = dt
        self.save_frames = save_frames
        self.visualizer = Visualizer()
        self.integrator = Integrator()

    def simulate(self, velocity_field, initial_positions, u_initial, domain, experiment_name, plot_every=5):
        x_min, x_max, y_min, y_max = domain
        n_particles = initial_positions.shape[0]
        n_steps = int(self.t_max / self.dt)

        # История положений и значений поля
        history = {
            't': np.zeros(n_steps + 1),
            'x': np.zeros((n_steps + 1, n_particles)),
            'y': np.zeros((n_steps + 1, n_particles)),
            'u': np.zeros((n_steps + 1, n_particles))
        }

        # Инициализация
        history['t'][0] = 0
        history['x'][0] = initial_positions[:, 0]
        history['y'][0] = initial_positions[:, 1]
        history['u'][0] = u_initial

        # Основной цикл интегрирования
        for step in range(n_steps):
            t = step * self.dt

            # Обновление положений частиц методом Рунге-Кутты
            for i in range(n_particles):
                x, y = history['x'][step, i], history['y'][step, i]
                x_new, y_new = self.integrator.rk4_step(x, y, t, self.dt, velocity_field)

                # Проверка границ (опционально)
                x_new = np.clip(x_new, x_min, x_max)
                y_new = np.clip(y_new, y_min, y_max)

                history['x'][step + 1, i] = x_new
                history['y'][step + 1, i] = y_new
                history['u'][step + 1, i] = history['u'][step, i]  # Значение u сохраняется вдоль траектории

            history['t'][step + 1] = t + self.dt

            # Визуализация через определенное количество шагов
            if step % plot_every == 0 or step == n_steps - 1:
                print(f"Шаг {step + 1}/{n_steps}, Время: {t + self.dt:.2f}")

                self.visualizer.plot_particles(
                    history['x'][step + 1], history['y'][step + 1], history['u'][step + 1],
                    t + self.dt, domain, experiment_name,
                    show_trajectories=True,
                    trajectories={
                        'x': history['x'][0:step + 1],
                        'y': history['y'][0:step + 1],
                        't': history['t'][0:step + 1],
                        'u': history['u'][0:step + 1]
                    },  # ограничиваем траектории
                    save=self.save_frames
                )

        # Визуализация траекторий всех частиц
        self.visualizer.plot_trajectories(history, domain, experiment_name, save=self.save_frames)

        return history


class Experiment:
    """Класс для проведения экспериментов"""

    def __init__(self, name, velocity_field, domain=(0, 10, 0, 10)):
        self.name = name
        self.velocity_field = velocity_field
        self.domain = domain
        self.visualizer = Visualizer()

    def run(self, simulation, initial_positions, u_initial):
        """
        Запускает эксперимент

        Параметры:
        simulation - объект класса LagrangianSimulation
        initial_positions - начальные позиции частиц
        u_initial - начальные значения поля u

        Возвращает:
        history - история положений частиц и значений поля
        """
        print(f"\n=== {self.name} ===")

        # Визуализация поля скоростей
        self.visualizer.plot_velocity_field(
            self.velocity_field, 0, self.domain,
            title=self.name,
            filename=f"{self.name.lower().replace(' ', '_')}_velocity_field.png"
        )

        # Моделирование
        history = simulation.simulate(
            self.velocity_field, initial_positions, u_initial,
            self.domain, self.name.lower().replace(' ', '_'),
            plot_every=20
        )

        return history


import time


def main():
    simulation = LagrangianSimulation(n_particles=1000, t_max=10.0, dt=0.05, save_frames=True)

    # Experiment 1: Вихревое течение
    experiment1 = Experiment("Эксперимент 1: Вихревое течение", VelocityField.vortex_flow)
    initial_positions = ParticleGenerator.radial_pattern(simulation.n_particles, x0=2.5, y0=5, r_max=2)[0]
    u_initial = InitialCondition.gaussian(initial_positions[:, 0], initial_positions[:, 1], 5, 5, 1.0)

    start_time = time.time()  # Start time for experiment 1
    history1 = experiment1.run(simulation, initial_positions, u_initial)
    end_time = time.time()  # End time for experiment 1
    print(f"Время выполнения {experiment1.name}: {end_time - start_time:.2f} секунд")

    # Experiment 2: Сдвиговое течение
    experiment2 = Experiment("Эксперимент 2: Сдвиговое течение", VelocityField.shear_flow)
    initial_positions = ParticleGenerator.grid_pattern(simulation.n_particles)
    u_initial = InitialCondition.circle(initial_positions[:, 0], initial_positions[:, 1], 5, 5, 2.0)

    start_time = time.time()  # Start time for experiment 2
    history2 = experiment2.run(simulation, initial_positions, u_initial)
    end_time = time.time()    # End time for experiment 2
    print(f"Время выполнения {experiment2.name}: {end_time - start_time:.2f} секунд")

    # Experiment 3: Течение с источником и стоком
    experiment3 = Experiment("Эксперимент 3: Течение с источником и стоком", VelocityField.source_sink_flow)
    initial_positions, u_initial = ParticleGenerator.radial_pattern(simulation.n_particles)

    start_time = time.time()  # Start time for experiment 3
    history3 = experiment3.run(simulation, initial_positions, u_initial)
    end_time = time.time()    # End time for experiment 3
    print(f"Время выполнения {experiment3.name}: {end_time - start_time:.2f} секунд")

    print("\nВсе эксперименты завершены. Изображения сохранены в директории 'lab6_images'.")


if __name__ == "__main__":
    main()

