# python
from typing import Callable

import numpy as np
from numpy import ndarray

from ..libs import AlgebraicSolver

class DifferentialEquationSolver:
    """
    A class for solving differential equations using different methods.

    Methods:
        fdm(...): Constructs the finite difference scheme from given differential operators and boundary conditions
                  and solves the resulting linear system.
    """

    def fdm(self,
            alpha_0: float,
            alpha_1: float,
            beta_0: float,
            beta_1: float,
            gamma_0: float,
            gamma_1: float,
            n: int,
            h: float,
            x: ndarray,
            p_x: Callable[[float | ndarray], float | ndarray],
            q_x: Callable[[float | ndarray], float | ndarray],
            f_x: Callable[[float | ndarray], float | ndarray],
            a_hr: Callable[[float | ndarray, float | ndarray], float | ndarray],
            c_hr: Callable[[float | ndarray, float | ndarray], float | ndarray]) -> ndarray:
        """
        Solves a differential equation using the finite difference method by constructing and solving
        the linear system A*y = B based on the discretized equation and boundary conditions.

        Args:
            alpha_0 (float): Coefficient for the left boundary condition.
            alpha_1 (float): Coefficient for the right boundary condition.
            beta_0 (float): Coefficient for the left boundary condition.
            beta_1 (float): Coefficient for the right boundary condition.
            gamma_0 (float): Constant for the left boundary condition.
            gamma_1 (float): Constant for the right boundary condition.
            n (int): Number of subintervals.
            h (float): Step size.
            x (ndarray): Array of x values.
            p_x (Callable): Function to compute p(x).
            q_x (Callable): Function to compute q(x).
            f_x (Callable): Function to compute f(x).
            a_hr (Callable): Function to compute the finite difference coefficient a.
            c_hr (Callable): Function to compute the finite difference coefficient c.

        Returns:
            ndarray: Solution vector y for the discretized differential equation.
        """
        r = p_x(x) * h / 2
        a_values = a_hr(h, r)
        c_values = c_hr(h, r)
        b_values = q_x(x) - a_values - c_values
        f_values = f_x(x)

        A = np.zeros((n + 1, n + 1))
        B = np.zeros(n + 1)

        for i in range(1, n):
            A[i][i - 1] = a_values[i]
            A[i][i] = b_values[i]
            A[i][i + 1] = c_values[i]
            B[i] = f_values[i]

        A[0, 0] = alpha_0 - beta_0 / h
        A[0, 1] = beta_0 / h
        B[0] = gamma_0

        A[n, n - 1] = -beta_1 / h
        A[n, n] = alpha_1 + beta_1 / h
        B[n] = gamma_1

        solver = AlgebraicSolver()
        y = solver.tdma(A, B)

        return y

    def runge_kutt_4(self,
                     x: ndarray,
                     y0: float,
                     h: float,
                     n: int,
                     f: Callable[[float, float, float, float, float], float],
                     a: float,
                     b: float,
                     c: float) -> ndarray:
        """
            Solves a differential equation using the 4th order Runge-Kutta method.

            Args:
                x (ndarray): Array of x values.
                y0 (float): Initial value of y.
                h (float): Step size.
                n (int): Number of steps.
                f (Callable): Function representing the differential equation.
                a (float): Parameter a of the equation.
                b (float): Parameter b of the equation.
                c (float): Parameter c of the equation.

            Returns:
                ndarray: Array of y values computed using the Runge-Kutta method.
            """
        y_values = np.array([y0], dtype=float)

        for i in range(n):
            x_current = float(x[i])
            y_current = float(y_values[-1])

            k1 = f(x_current, y_current, a, b, c)
            k2 = f(x_current + h / 2, y_current + h / 2 * k1, a, b, c)
            k3 = f(x_current + h / 2, y_current + h / 2 * k2, a, b, c)
            k4 = f(x_current + h, y_current + h * k3, a, b, c)

            y_values = np.append(y_values, y_current + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

        return y_values
