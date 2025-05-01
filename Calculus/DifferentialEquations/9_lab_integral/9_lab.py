import numpy as np
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt

N     = 100
λ     = 1.0
sgn   = +1
a, b  = 0.0, 1.0

x = np.linspace(a, b, N)
h = (b - a) / (N - 1)
w = np.full(N, h)
w[0]  = h / 2
w[-1] = h / 2

def kernel(x, s):
    return (1 - s) * (np.exp(0.2 * x * s) - 1)

Kmat = kernel(x[:, None], x[None, :]) * w

A = np.eye(N) - sgn * λ * Kmat
f = 1 + x
u = solve(A, f)

residual = norm(u - (f + sgn * λ * Kmat @ u))
print(f"норма невязки = {residual:.3e}")

plt.plot(x, u, label='u(x)', color='blue')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Решение интегрального уравнения')
plt.grid(True)
plt.legend()
plt.show()
