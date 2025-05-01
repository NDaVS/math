import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


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
    k = c * dt / (dx ** 2)

    for ti in range(m):
        for j in range(1, n):
            u_[ti + 1, j] = u_[ti, j] + k * (u_[ti, j + 1] - 2 * u_[ti, j] + u_[ti, j - 1])

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

def animate_solution(x, values, dt, name, save_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, values[0], lw=2)
    ax.set_ylim(np.min(values), np.max(values))
    ax.set_title(name)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.grid()

    def update(frame):
        line.set_ydata(values[frame])  # Update the y-data of the line
        ax.set_title(f"{name} — t = {frame * dt:.4f}")  # Update the title with time
        return line,

    step = max(1, len(values) // 300)  # Adjust step size for smoother animation
    frames = range(0, len(values), step)  # Define frames to iterate over

    anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)  # Enable blitting for performance

    # Save the animation as a GIF
    anim.save(f"{save_name}.gif", writer=PillowWriter(fps=30))
    plt.close()

def main():
    L = 10
    c = 3
    Nx = 100
    T = L / c

    dt = 0.0001
    Nt = int(T / dt) + 1
    dx = L / Nx

    x = np.linspace(0, L, Nx + 1, endpoint=False)
    CFL = c * dt / dx
    if CFL > 1:
        raise ValueError(f"Схема неустойчива: CFL = {CFL:.2f} > 1")

    def get_initial_u():
        u = np.zeros((Nt + 1, Nx + 1))
        u[0] = base_function(x)
        u[:, 0] = ux0(0)
        u[:, -1] = uxl(0)
        return u

    print("Считаем метод 'Вверх по потоку'...")
    u = get_initial_u()
    values_upstream = upstream(u, c, dx, dt)
    animate_solution(x, values_upstream, dt, "Метод вверх по потоку", "upstream")

    print("Считаем явную схему...")
    u = get_initial_u()
    values_explicit = explicit(u, c, dx, dt)
    animate_solution(x, values_explicit, dt, "Явная схема", "explicit")

    print("Считаем неявную схему...")
    u = get_initial_u()
    values_implicit = implicit(u, c, dx, dt)
    animate_solution(x, values_implicit, dt, "Неявная схема", "implicit")

    print("GIF-анимации сохранены: upstream.gif, explicit.gif, implicit.gif")

if __name__ == '__main__':
    main()
