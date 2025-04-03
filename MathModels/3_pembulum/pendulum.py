import numpy as np
import matplotlib.pyplot as plt

g = 9.81
l = 1.0
b = 0.1
m = 1.0
F = 0
# omega_f = np.sqrt(g/l)/3
omega_f = 1
theta0 = np.pi / 10
omega0 = 0

def pendulum_eq_lin(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / l) * theta - (b / m) * omega + F * np.sin(omega_f * t)
    return np.array([dtheta_dt, domega_dt])

def pendulum_eq(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta) - (b / m) * omega + F * np.sin(omega_f * t)
    return np.array([dtheta_dt, domega_dt])


def runge_kutta4(f, y0, t_span, dt):
    t_values = np.arange(t_span[0], t_span[1] + dt, dt)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]

        k1 = dt * f(t, y)
        k2 = dt * f(t + dt / 2, y + k1 / 2)
        k3 = dt * f(t + dt / 2, y + k2 / 2)
        k4 = dt * f(t + dt, y + k3)

        y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values.T

t_span = (0, 25)
dt = 0.025

t_vals, sol  = runge_kutta4(pendulum_eq, [theta0, omega0], t_span, dt)
t_vals_lin, sol_lin = runge_kutta4(pendulum_eq_lin, [theta0, omega0], t_span, dt)


def draw(t_vals, theta_vals, omega_vals):
    plt.figure(figsize=(12, 6))
    split_idx = len(t_vals) // 3

    plt.subplot(1, 2, 1)
    plt.plot(t_vals, theta_vals, 'b', label=r'$\theta(t)$')
    plt.plot(t_vals[0], theta_vals[0], 'go', label='Начальное положение')
    plt.xlabel('Время, с')
    plt.ylabel('Угол (рад)')
    plt.title('Динамика математического маятника')
    plt.legend(loc='upper right')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(theta_vals, omega_vals, 'b',
             label=r'Фазовый портрет ($\omega$ от $\theta$)')
    plt.plot(theta_vals[0], omega_vals[0], 'go', label='Начальное положение')
    plt.title('Фазовый портрет математического маятника')
    plt.xlabel('Угол (рад)')
    plt.ylabel('Угловая скорость (рад/с)')
    plt.legend(loc='upper right')
    plt.grid()

    plt.tight_layout()
    plt.show()

# draw(t_vals, sol[0], sol[1])
draw(t_vals_lin, sol_lin[0], sol_lin[1])


def get_amplitude(f_eq, omega_ff, muu, t_spann, dtt):
    global b, F, omega_f
    b = muu  # изменяем коэффициент трения
    omega_f = omega_ff
    t_vals, sol = runge_kutta4(f_eq, [theta0, omega0], t_spann, dtt)
    theta_vals = sol[0]
    last_quarter = theta_vals[len(theta_vals) * 3 // 4:]
    return np.max(np.abs(last_quarter))


# Определяем диапазоны частот
omega = np.sqrt(g / l)
omega_f_values_1 = np.linspace(0.5 * omega, 2 * omega, 60)
omega_f_values_2 = np.linspace(0.9 * omega, 1.1 * omega, 60)

mu_values = [1, 0.75, 0.5]  # Разные коэффициенты трения


def calculate_theta(F_ee, mm,  omegaa, muu):
    # Вычисление коэффициента затухания ζ
    zeta = muu / (2 * mm * omegaa)
    omega_fff = omegaa  * np.sqrt(1 - 2 * zeta ** 2)
    # Вычисление θ(ω_f)
    theta = F_ee / (mm * np.sqrt((2 * omega_fff * omegaa * zeta) ** 2 + (omegaa ** 2 - omega_fff ** 2) ** 2))
    return omega_fff, theta
plt.figure(figsize=(12, 6))

for mu in mu_values:
    amplitudes_1 = [get_amplitude(pendulum_eq_lin, wf, mu, t_span, dt) for wf in omega_f_values_1]
    amplitudes_2 = [get_amplitude(pendulum_eq_lin, wf, mu, t_span, dt) for wf in omega_f_values_2]
    max_amplitude_index_1 = np.argmax(amplitudes_1)
    max_amplitude_index_2 = np.argmax(amplitudes_2)

    omega_f_ , theta = calculate_theta(F, m, omega, mu)


    plt.subplot(1, 2, 1)
    plt.plot(omega_f_values_1 / omega, amplitudes_1, label=f'μ={mu}, [0.5w, 2w]')
    plt.plot(omega_f_/ omega, theta, 'o',markersize=8, label=f'Теор. макс. μ={mu}')
    plt.plot(omega_f_values_1[max_amplitude_index_1] / omega, amplitudes_1[max_amplitude_index_1], 'o',
             label=f'Факт. макс. μ={mu}')
    plt.title('Резонансные кривые')

    plt.xlabel('Отношение $\\omega_f / \\omega$')
    plt.ylabel('Амплитуда $\\theta_{max}$')
    plt.title('Резонансные кривые')
    plt.legend()
    plt.grid()



    plt.subplot(1, 2, 2)
    plt.plot(omega_f_values_2 / omega, amplitudes_2, label=f'μ={mu}, [0.9w, 1.1w]')
    plt.plot(omega_f_/ omega, theta, 'o',markersize=8 , label=f'Теор. макс. μ={mu}')
    plt.plot(omega_f_values_2[max_amplitude_index_2] / omega , amplitudes_2[max_amplitude_index_2], 'o',
             label=f'Факт. макс. μ={mu}')
    plt.xlabel('Отношение $\\omega_f / \\omega$')
    plt.ylabel('Амплитуда $\\theta_{max}$')
    plt.title('Резонансные кривые')
    plt.legend()
    plt.grid()

plt.show()