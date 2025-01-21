from collections import Counter
import numpy as np


def hurwitz_matrix(coefficients):
    n = len(coefficients) - 1  # Степень полинома
    hurwitz = np.zeros((n, n))  # Создаем пустую квадратную матрицу n x n

    for i in range(n):
        for j in range(n):
            # Индекс в списке коэффициентов
            index = 2 * i - j + 1
            if 0 <= index < len(coefficients):
                hurwitz[i, j] = coefficients[index]

    return hurwitz


def calculate_principal_minors(matrix):
    minors = []
    for k in range(1, len(matrix) + 1):
        sub_matrix = matrix[:k, :k]  # Выбираем ведущую подматрицу размера k x k
        determinant = np.linalg.det(sub_matrix)  # Вычисляем определитель
        minors.append(determinant)
    return minors


# Пример использования
coeffs =[1, 2, 2, 3]
hurwitz = hurwitz_matrix(coeffs)
print("Матрица Гурвица:")
print(hurwitz)

# Нахождение угловых миноров
minors = calculate_principal_minors(hurwitz)
print("Угловые миноры:")
for i, minor in enumerate(minors, start=1):
    print(f"Δ{i} = {minor}")


print("Критерий гурвица: ", all(minor > 0 for minor in minors))
import numpy as np
import matplotlib.pyplot as plt




def mikhailov_criterion(coefficients):
    # Получаем корни характеристического уравнения
    roots = np.roots(coefficients)

    # Проверяем, находятся ли корни в левой полуплоскости
    for root in roots:
        if np.real(root) >= 0:
            return False
    return True


def visualize_mikhailov(coefficients):
    # Получаем корни характеристического уравнения
    roots = np.roots(coefficients)

    # Вещественные части всех корней
    real_parts = [root.real for root in roots]

    # Учитываем количество одинаковых вещественных частей
    counts = Counter(real_parts)
    sizes = [50 + 30 * (counts[real] - 1) for real in real_parts]  # Увеличиваем размер для повторений

    # Генерация цветов для каждой точки
    colors = plt.cm.rainbow(np.linspace(0, 1, len(real_parts)))

    # Визуализация вещественных частей корней
    plt.figure(figsize=(8, 6))

    # Отображаем все вещественные части
    for real_part, size, color in zip(real_parts, sizes, colors):
        plt.scatter(real_part, 0, color=color, s=size, label=f'{real_part:.2f}')

    # Отметим оси
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')

    # Настройки графика
    plt.title('Вещественные части корней с разными размерами для одинаковых точек')
    plt.xlabel('Действительная часть')
    plt.ylabel('Мнимая часть')
    plt.xlim(min(real_parts) - 1, max(real_parts) + 1)
    plt.ylim(-1, 1)
    plt.grid()
    plt.legend(loc='upper left', fontsize='small', frameon=True)
    plt.show()


# Пример использования
print("Критерий Михайлова:", mikhailov_criterion(coeffs))
visualize_mikhailov(coeffs)


def lienard_shipar_criterion(coefficients):
    # Примерная реализация, основанная на анализе корней
    roots = np.roots(coefficients)
    real_parts = np.real(roots)
    imag_parts = np.imag(roots)

    # Проверяем условия устойчивости
    if all(real_parts < 0):
        return True
    return False


# Пример использования
print("Критерий Льенара-Шипара:", lienard_shipar_criterion(coeffs))
