import numpy as np
import matplotlib.pyplot as plt
from ldap3.protocol.formatters.formatters import format_integer


def base_function(x):
    return np.exp(-2 * (x - 0.3) **4)


def ux0(t):
    return 0


def uxl(t):
    return 0


def explicit(uu, c, dx, dt):
    u_ = uu.copy()
    k = c * dt / (1 *dx)
    for ti in range(u_.shape[0]-1):
        for xi in range(1, u_.shape[1]-1):
            u_[ti + 1, xi] = u_[ti, xi] + k * (u_[ti, xi - 1] - u_[ti, xi + 1])
    return u_


def upstream(uu,c, dx, dt):
    u_ = uu.copy()
    k = c * dt / dx
    for ti in range(u_.shape[0]-1):
        for xi in range(1, u_.shape[1] -1):
            u_[ti + 1, xi] = u_[ti, xi] + k * (u_[ti, xi - 1] - u_[ti, xi])

    return u_

def TDMA(a,b,c,f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, c, b, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0]*n

    for i in range(1, n):
        alpha.append(-b[i]/(a[i]*alpha[i-1] + c[i]))
        beta.append((f[i] - a[i]*beta[i-1])/(a[i]*alpha[i-1] + c[i]))

    x[n-1] = beta[n - 1]

    for i in range(n-1, -1, -1):
        x[i - 1] = alpha[i - 1]*x[i] + beta[i - 1]

    return x

def implicit(uu,c, dx, dt):
    m = uu.shape[0] - 1
    n = uu.shape[1] - 1
    u_ = uu.copy()
    k = c * dt / (2 * dx)
    for ti in range(m):
        al = np.zeros(n + 1)
        bl = np.zeros(n + 1)
        cl = np.zeros(n + 1)

        al[1:-1] = -k
        bl[:] = 1
        cl[1:-1] = k

        bl[0] = 1
        bl[-1] = 1

        u_[ti + 1] = TDMA(al, bl, cl, u_[ti])

    return u_
def main():
    # Параметры задачи
    L = 100.0
    c = 1
    Nx = 1000
    T = L / c
    dt = 0.01
    Nt = int(T / dt) + 1

    dx = L / Nx

    x = np.linspace(0, L, Nx + 1, endpoint=False)
    CFL = c * dt / dx

    if CFL > 1:
        raise ValueError(f"Схема неустойчива: CFL = {CFL:.2f} > 1")

    u = np.zeros((Nt + 1, Nx + 1))
    u[0] = base_function(x)
    u[:, 0] = ux0(0)
    u[:, -1] = uxl(0)
    values = upstream(u, c, dx, dt)
    plt.figure(figsize=(12, 8))
    for i in range(0, len(values)-50, len(values) // 5):
        plt.plot(x, values[i], label=f't = {i * dt:.2f}')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
