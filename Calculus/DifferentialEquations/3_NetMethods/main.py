import numpy as np
from matplotlib import pyplot as plt

from Calculus.DifferentialEquations.libs import AlgebraicSolver


def phi(x, t):
    """
    Function representing the source term in the heat equation.

    Parameters:
    -----------
    x : float
        Spatial coordinate.
    t : float
        Time coordinate.

    Returns:
    --------
    float
        Value of the source term at (x, t).
    """
    return x ** 2 * (1 - x)


def psi(x):
    """
    Initial condition function for the heat equation.

    Parameters:
    -----------
    x : float
        Spatial coordinate.

    Returns:
    --------
    float
        Initial temperature distribution at x.
    """
    return x * (x ** 2 - x)  # Исправлен оператор ^ на **


def explicit_scheme(a, gamma_0, gamma_1, M, N, l, T):
    """
    Solves the heat equation using the explicit finite difference scheme.

    Parameters:
    -----------
    a : float
        Thermal diffusivity.
    gamma_0 : float
        Boundary condition at x=0.
    gamma_1 : float
        Boundary condition at x=l.
    M : int
        Number of spatial grid points.
    N : int
        Number of time steps.
    l : float
        Length of the spatial domain.
    T : float
        Total time.

    Returns:
    --------
    tuple
        Arrays of spatial coordinates, time coordinates, and temperature values.
    """
    h = l / M
    tau = T / N
    lambda_ = (a ** 2 * tau) / (h ** 2)
    print(lambda_)
    if lambda_ > 0.5:
        print(
            f"Warning: Explicit scheme may be unstable (lambda={lambda_:.3f} > 0.5). Consider reducing tau or increasing M.")

    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)
    u = np.zeros((N + 1, M + 1))

    # Начальное условие
    for m in range(M + 1):
        u[0, m] = psi(x[m])

    # Граничные условия
    for n in range(N + 1):
        u[n, 0] = gamma_0
        u[n, M] = gamma_1

    # Явная разностная схема
    for n in range(N):
        for m in range(1, M):
            u[n + 1, m] = (
                    u[n, m] + lambda_ * (u[n, m + 1] - 2 * u[n, m] + u[n, m - 1]) + tau * phi(x[m], t[n])
            )

    return x, t, u


def implicit_scheme(a, gamma_0, gamma_1, M, N, l, T):
    """
    Solves the heat equation using the implicit finite difference scheme.

    Parameters:
    -----------
    a : float
        Thermal diffusivity.
    gamma_0 : float
        Boundary condition at x=0.
    gamma_1 : float
        Boundary condition at x=l.
    M : int
        Number of spatial grid points.
    N : int
        Number of time steps.
    l : float
        Length of the spatial domain.
    T : float
        Total time.

    Returns:
    --------
    tuple
        Arrays of spatial coordinates, time coordinates, and temperature values.
    """
    h = l / M
    tau = T / N
    lambda_ = (a ** 2 * tau) / (h ** 2)

    x = np.linspace(0, l, M + 1)
    t = np.linspace(0, T, N + 1)
    u = np.zeros((N + 1, M + 1))

    # Начальное условие
    for m in range(M + 1):
        u[0, m] = psi(x[m])

    # Граничные условия
    for n in range(N + 1):
        u[n, 0] = gamma_0
        u[n, M] = gamma_1

    # Неявная схема (метод прогонки)
    A = -lambda_ * np.ones(M - 1)
    B = (1 + 2 * lambda_) * np.ones(M - 1)
    C = -lambda_ * np.ones(M - 1)

    for n in range(N):
        d = u[n, 1:M] + tau * phi(x[1:M], t[n])
        d[0] += lambda_ * gamma_0
        d[-1] += lambda_ * gamma_1
        u[n + 1, 1:M] = initiate_tdma(A, B, C, d)

    return x, t, u


def initiate_tdma(A, B, C, d):
    """
    Solves a tridiagonal system of linear equations using the Thomas algorithm.

    Parameters:
    -----------
    A : array_like
        Sub-diagonal elements.
    B : array_like
        Diagonal elements.
    C : array_like
        Super-diagonal elements.
    d : array_like
        Right-hand side vector.

    Returns:
    --------
    array_like
        Solution vector.
    """
    solver = AlgebraicSolver()
    n = len(B)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, B)
    np.fill_diagonal(matrix[1:], A)
    np.fill_diagonal(matrix[:, 1:], C)
    return solver.tdma(matrix, d)


def plot_solution(x, t, u, title):
    """
    Plots the solution of the heat equation.

    Parameters:
    -----------
    x : array_like
        Spatial coordinates.
    t : array_like
        Time coordinates.
    u : array_like
        Temperature values.
    title : str
        Title of the plot.
    """
    X, T = np.meshgrid(x, t)
    plt.figure(figsize=(8, 6))
    plt.contourf(X, T, u, 20, cmap="hot")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.show()


def main():
    """
    Main function to solve and plot the heat equation using explicit and implicit schemes.
    """
    a = 0.001
    gamma_0 = gamma_1 = 0
    M = 1000
    N = 6
    l = 1
    T = 2

    x, t, u_explicit = explicit_scheme(a, gamma_0, gamma_1, M, N, l, T)
    plot_solution(x, t, u_explicit, "Heat Equation Solution (Explicit Scheme)")

    x, t, u_implicit = implicit_scheme(a, gamma_0, gamma_1, M, N, l, T)
    plot_solution(x, t, u_implicit, "Heat Equation Solution (Implicit Scheme)")


if __name__ == "__main__":
    main()