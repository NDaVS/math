import matplotlib.pyplot as plt
import numpy as np


def base_function(x):
    return 4 * np.exp(-100 * (x) ** 4)


def ux0(t):
    return 0


def uxl(t):
    return 0


def explicit(uu, c, dx, dt):
    m = uu.shape[0] - 1
    n = uu.shape[1] - 1
    u_ = uu.copy()
    k = c * dt / (2 * dx)

    for ti in range(m):
        for j in range(1, n):
            u_[ti + 1, j] = u_[ti, j] - k * (u_[ti, j + 1] - u_[ti, j - 1])

        # Граничные условия (можно поставить 0, или Neumann, если нужно)
        u_[ti + 1, 0] = u_[ti, 0]
        u_[ti + 1, n] = u_[ti, n]

    return u_


def upstream(uu, c, dx, dt):
    u_ = uu.copy()
    k = c * dt / dx
    for ti in range(u_.shape[0] - 1):
        for xi in range(1, u_.shape[1] - 1):
            u_[ti + 1, xi] = u_[ti, xi] + k * (u_[ti, xi - 1] - u_[ti, xi])
    return u_


def TDMA(a, b, c, f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, c, b, f)))
    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0] * n
    for i in range(1, n):
        denom = a[i] * alpha[i - 1] + c[i]
        alpha.append(-b[i] / denom)
        beta.append((f[i] - a[i] * beta[i - 1]) / denom)
    x[n - 1] = beta[n - 1]
    for i in range(n - 1, 0, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]
    return x


def implicit(uu, c, dx, dt):
    m = uu.shape[0] - 1
    n = uu.shape[1] - 1
    u_ = uu.copy()
    k = c * dt / (2 * dx)
    for ti in range(m):
        al = np.zeros(n + 1)
        bl = np.ones(n + 1)
        cl = np.zeros(n + 1)
        al[1:-1] = -k
        cl[1:-1] = k
        bl[0] = bl[-1] = 1
        u_[ti + 1] = TDMA(al, bl, cl, u_[ti])
    return u_


def make_plot(x, u):
    plt.figure(figsize=(12, 8))

    plt.plot(x, u[0], label='Положение волны в начальный момент времени')
    plt.plot(x, u[int(u.shape[0] * 0.1)], label='Положение волны через 10% времени')
    plt.plot(x, u[int(u.shape[0] * 0.3)], label='Положение волны через 30% времени')
    plt.plot(x, u[int(u.shape[0] * 0.5)], label='Положение волны через 50% времени')
    plt.plot(x, u[int(u.shape[0] * 0.8)], label='Положение волны через 80% времени')

    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_integrals(x, u):
    # Вычисление интегралов по x при фиксированных t
    t0 = 0
    t10 = int(u.shape[0] * 0.1)
    t30 = int(u.shape[0] * 0.3)
    t50 = int(u.shape[0] * 0.5)
    t80 = int(u.shape[0] * 0.8)

    I0 = np.trapezoid(u[t0], x)
    I10 = np.trapezoid(u[t10], x)
    I30 = np.trapezoid(u[t30], x)
    I50 = np.trapezoid(u[t50], x)
    I80 = np.trapezoid(u[t80], x)

    return {
        't = 0%': I0,
        't = 10%': I10,
        't = 30%': I30,
        't = 50%': I50,
        't = 80%': I80,
    }


def main():
    L = 5
    c = 1
    Nx = 100
    T = L / c

    dt = 0.001 / 2
    Nt = int(T / dt) + 1
    dx = L / Nx
    print(c * dt / (dx * 2))
    x = np.linspace(0, L, Nx + 1, endpoint=False)
    CFL = c * dt / dx
    print(CFL)

    def get_initial_u():
        u = np.zeros((Nt + 1, Nx + 1))
        u[0] = base_function(x)
        u[:, 0] = ux0(0)
        u[:, -1] = uxl(0)
        return u

    u = get_initial_u()
    values_upstream = upstream(u, c, dx, dt)
    values_explicit = explicit(u, c, dx, dt)
    values_implicit = implicit(u, c, dx, dt)
    make_plot(x, values_upstream)
    make_plot(x, values_explicit)
    make_plot(x, values_implicit)

    integrals = compute_integrals(x, values_upstream)
    print(integrals)
    integrals = compute_integrals(x, values_explicit)
    print(integrals)
    integrals = compute_integrals(x, values_implicit)
    print(integrals)


if __name__ == '__main__':
    main()
