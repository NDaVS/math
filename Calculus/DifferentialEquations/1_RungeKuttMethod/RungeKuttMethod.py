import numpy as np
from prettytable import PrettyTable

from Calculus.DifferentialEquations.libs import DifferentialEquationSolver


def true_f(x: float, a: float, b: float, c: float) -> float:
    """
    Analytical solution of the differential equation.

    Args:
        x (float): The independent variable.
        a (float): Parameter a of the equation.
        b (float): Parameter b of the equation.
        c (float): Parameter c of the equation.

    Returns:
        float: The analytical solution at x.
    """
    return a / b - (c - x) ** b * a / (c ** b * b)

def f(x: float, y: float, a: float, b: float, c: float) -> float:
    """
    Differential equation function.

    Args:
        x (float): The independent variable.
        y (float): The dependent variable.
        a (float): Parameter a of the equation.
        b (float): Parameter b of the equation.
        c (float): Parameter c of the equation.

    Returns:
        float: The value of the function at (x, y).
    """
    return (a - b * y) / (c - x)

def solve(x0, xn, y0, n, a, b, c):
    """
    Solves the differential equation using the 4th order Runge-Kutta method and prints the results.

    Args:
        x0 (float): Initial value of x.
        xn (float): Final value of x.
        y0 (float): Initial value of y.
        n (int): Number of steps.
        a (float): Parameter a of the equation.
        b (float): Parameter b of the equation.
        c (float): Parameter c of the equation.
    """
    h: float = (xn - x0) / n

    x = np.linspace(x0, xn, n)

    dif_solver = DifferentialEquationSolver()
    y_values_rk = dif_solver.runge_kutt_4(x, y0, h, n, f, a, b, c)

    # Compute analytical values and errors
    y_analytical_values = [true_f(xs, a, b, c) for xs in x]
    errors = [abs(y_analytical - y_rk) for y_analytical, y_rk in zip(y_analytical_values, y_values_rk)]

    table = PrettyTable()
    table.field_names = ["x", "Истинное значение", "Рунге-Кутта", "Ошибка"]

    for i in range(len(x)):
        table.add_row([f"{x[i]:.8f}", f"{y_analytical_values[i]:.12f}", f"{y_values_rk[i]:.12f}", f"{errors[i]:.12f}"])

    print(table)

def main():
    """
    Main function to solve the differential equation with given parameters and print the results.
    """
    a: float = 2
    b: float = 3
    c: float = 2
    x0: float = 0
    xn: float = 1
    y0: float = 0
    n: int = 100
    solve(x0, xn, y0, n, a, b, c)
    print('Уменьшим шаг в два раза')
    solve(x0, xn, y0, n * 2, a, b, c)

if __name__ == '__main__':
    main()