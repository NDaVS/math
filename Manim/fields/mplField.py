import numpy as np
import matplotlib.pyplot as plt

# Создаем сетку для значений x и y
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# Определяем производные
U = Y  # x' = y
V = -X  # y' = -x

# Рисуем поле векторов
plt.quiver(X, Y, U, V, color='r')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Фазовое поле для системы x\' = y, y\' = -x')
plt.grid()
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.show()
