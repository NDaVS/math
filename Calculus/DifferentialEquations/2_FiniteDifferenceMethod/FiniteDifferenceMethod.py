import numpy as np
from numpy import ndarray
from prettytable import PrettyTable

from Calculus.DifferentialEquations.libs import DifferentialEquationSolver


def p(x: float | ndarray) -> float | ndarray:
    """
    Compute the value of p(x).

    Args:
        x (float | ndarray): Input value or array.

    Returns:
        float | ndarray: The computed p(x) = 4*x / (x^2 + 1).
    """
    return 4 * x / (x ** 2 + 1)


def q(x: float | ndarray) -> float | ndarray:
    """
    Compute the value of q(x).

    Args:
        x (float | ndarray): Input value or array.

    Returns:
        float | ndarray: The computed q(x) = -1 / (x^2 + 1).
    """
    return -1 / (x ** 2 + 1)


def f(x: float | ndarray) -> float | ndarray:
    """
    Compute the value of f(x).

    Args:
        x (float | ndarray): Input value or array.

    Returns:
        float | ndarray: The computed f(x) = -3 / ((x^2 + 1)^2).
    """
    return -3 / ((x ** 2 + 1) ** 2)


def true_function(x: float | ndarray) -> float | ndarray:
    """
    Compute the true analytical solution.

    Args:
        x (float | ndarray): Input value or array.

    Returns:
        float | ndarray: The true function value 1 / (x^2 + 1).
    """
    return 1 / (x ** 2 + 1)


def a(h: float | ndarray, r: float | ndarray) -> float | ndarray:
    """
    Compute the coefficient a for the finite difference scheme.

    Args:
        h (float | ndarray): Step size.
        r (float | ndarray): Parameter computed from p(x) and h.

    Returns:
        float | ndarray: The computed coefficient a.
    """
    return (1 + (r ** 2) / (1 + np.abs(r)) - r) / (h ** 2)


def c(h: float | ndarray, r: float | ndarray) -> float | ndarray:
    """
    Compute the coefficient c for the finite difference scheme.

    Args:
        h (float | ndarray): Step size.
        r (float | ndarray): Parameter computed from p(x) and h.

    Returns:
        float | ndarray: The computed coefficient c.
    """
    return (1 + (r ** 2) / (1 + np.abs(r)) + r) / (h ** 2)


def main():
    """
    Main function to solve the boundary value problem using the finite difference method.

    The function defines the differential equation parameters, calls the solver,
    computes the true solution, calculates errors, and prints the results in a table.
    """
    alpha_0: float = 0
    alpha_1: float = 1
    beta_0: float = 1
    beta_1: float = 0
    gama_0: float = 0
    gama_1: float = 0.5
    n: int = 100
    h: float = 1 / n
    x: ndarray = np.linspace(0, 1, n + 1)

    dif_solver = DifferentialEquationSolver()
    y = dif_solver.fdm(alpha_0, alpha_1, beta_0, beta_1, gama_0, gama_1, n, h, x, p, q, f, a, c)

    y_true = true_function(x)
    errors = y - y_true

    table = PrettyTable()
    table.field_names = ["x", "True Value", "Three-Point Method", "Error"]

    for i in range(len(x)):
        table.add_row([round(x[i], 5), round(y_true[i], 5), round(y[i], 5), round(errors[i], 5)])

    print(table)


if __name__ == '__main__':
    main()
