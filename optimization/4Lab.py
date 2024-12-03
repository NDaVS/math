import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # Для интерактивных графиков

class Gradient:

    def __init__(self, A: np.array, b: np.array, x0: np.array, alpha: float, h: float, max_iter=100000):
        self.max_iter = max_iter
        self.x = None
        self.alpha = alpha
        self.A = A
        self.b = b
        self.x0 = x0
        self.h = [h] * 3
        self.trajectory = []

    def f(self, x):
        return 0.5 * np.dot(x.T, np.dot(self.A, x)) + np.dot(self.b, x)

    def is_positive_definite(self, matrix):
        if matrix.shape[0] != matrix.shape[1]:
            return False
        for i in range(1, matrix.shape[0] + 1):
            minor = matrix[:i, :i]
            if np.linalg.det(minor) <= 0:
                return False
        return True

    def gradient_descent(self, tol=1e-6, max_iter=10000):
        if not self.is_positive_definite(self.A):
            raise ValueError("Matrix is not positive definite")

        x = self.x0
        self.trajectory = [x.copy()]  # Track trajectory

        for i in range(max_iter):
            grad = np.array([
                0.5 * (self.A[0][0] * x[0] * 2 + (self.A[0][1] + self.A[1][2]) * x[1] + (self.A[2, 0] + self.A[0, 2]) *
                       x[2]) + self.b[0],
                0.5 * (self.A[1, 1] * x[1] * 2 + (self.A[0][1] + self.A[1][2]) * x[0] + (self.A[1, 0] + self.A[2, 1]) *
                       x[2]) + self.b[1],
                0.5 * (self.A[2, 2] * x[2] * 2 + (self.A[2, 0] + self.A[0, 2]) * x[0] + (self.A[1, 0] + self.A[2, 1]) *
                       x[1]) + self.b[2]])
            x_new = x - self.alpha * grad
            self.trajectory.append(x_new.copy())  # Track trajectory

            if self.f(x_new) > self.f(x):
                self.alpha /= 2

            if np.linalg.norm(grad) < tol:
                break

            x = x_new

        return x, self.f(x)

    def check_h(self, x: np.array, i: int):
        old_f = self.f(x)
        local_h = self.h

        while True:
            x[i] = x[i] + local_h[i]

            if old_f > self.f(x):
                self.x = x
                return

            x[i] -= 2 * local_h[i]

            if old_f > self.f(x):
                self.x = x
                return

            x[i] += local_h[i]
            local_h[i] /= 2

    def partition_descent(self):
        x = self.x0
        self.trajectory = [x.copy()]  # Track trajectory

        for i in range(self.max_iter):
            old_f = self.f(x)
            for i in range(len(x)):
                self.check_h(x, i)
                self.trajectory.append(self.x.copy())  # Track trajectory
            if np.abs(old_f - self.f(self.x)) < self.alpha:
                return self.x, self.f(self.x)


def plot_interactive_3d(trajectory, method_name):
    trajectory = np.array(trajectory)
    fig = go.Figure()

    # Add a 3D scatter plot for the trajectory
    fig.add_trace(go.Scatter3d(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        z=trajectory[:, 2],
        mode='markers+lines',
        marker=dict(size=5),
        line=dict(color='blue', width=2),
        name=method_name
    ))

    # Set plot layout
    fig.update_layout(
        title=f"{method_name} - Interactive 3D Trajectory",
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='x3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


def plot_static_projections(trajectory, method_name):
    trajectory = np.array(trajectory)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Projection on x1-x2
    axes[0].plot(trajectory[:, 0], trajectory[:, 1], marker='o', label=f"{method_name} - x1-x2")
    axes[0].set_title("Projection on x1-x2")
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].legend()

    # Projection on x2-x3
    axes[1].plot(trajectory[:, 1], trajectory[:, 2], marker='o', label=f"{method_name} - x2-x3")
    axes[1].set_title("Projection on x2-x3")
    axes[1].set_xlabel('x2')
    axes[1].set_ylabel('x3')
    axes[1].legend()

    # Projection on x1-x3
    axes[2].plot(trajectory[:, 0], trajectory[:, 2], marker='o', label=f"{method_name} - x1-x3")
    axes[2].set_title("Projection on x1-x3")
    axes[2].set_xlabel('x1')
    axes[2].set_ylabel('x3')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    A = np.array([[2, 3, 1],
                  [2, 7, 2],
                  [1, 3, 3]])
    b = np.array([3, 4, 5])
    x0 = np.array([1, 1, 1], dtype='float')
    alpha = 0.01
    h = 0.2
    gradient = Gradient(A, b, x0, alpha, h)

    # Gradient Descent
    x_min, f_min = gradient.gradient_descent()
    print(f"Gradient Descent.\nMinimum at: {x_min}, Function value: {f_min}\n")
    plot_interactive_3d(gradient.trajectory, "Gradient Descent")
    plot_static_projections(gradient.trajectory, "Gradient Descent")

    # Partition Descent
    x_min, f_min = gradient.partition_descent()
    print(f"Partition Descent.\nMinimum at: {x_min}, Function value: {f_min}\n")
    plot_interactive_3d(gradient.trajectory, "Partition Descent")
    plot_static_projections(gradient.trajectory, "Partition Descent")


if __name__ == '__main__':
    main()
