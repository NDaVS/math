import numpy as np

from calculus.third_course.support.Rotation import RotationWithBarriers
from calculus.third_course.support.checker import check_answer


class Richardson:
    def __init__(self, A: np.array, b: np.array, x: np.array, p: int = 5):
        self._tol = 10 ** -(p - 1)  # Точность
        self._eta = None  # Отношение собственных значений
        self._tau_0 = None  # Оптимальное значение шага
        self._rho_0 = None  # Коэффициент сходимости
        self._A = A  # Матрица системы
        self._b = b  # Правая часть
        self._lambda_min = None  # Минимальное собственное значение
        self._lambda_max = None  # Максимальное собственное значение
        self._n = 8  # Количество узлов Чебышёва
        self._x = x  # Начальное приближение
        self._p = p  # Количество знаков точности

    def _compute_SZ(self):
        """Вычисление спектра матрицы (макс. и мин. собств. значений)."""
        rotation = RotationWithBarriers(self._A, self._p)
        # eigenvalues = np.linalg.eigvalsh(self._A)  # Используем эффективный метод для симметричных матриц
        eigenvalues = rotation.compute()
        # Оставляем только положительные собственные значения
        positive_eigenvalues = [val for val in eigenvalues if val > 0]

        if not positive_eigenvalues:  # Проверяем, есть ли положительные значения
            raise ValueError("Нет положительных собственных значений в спектре матрицы.")

        # Находим максимальное и минимальное среди положительных значений
        self._lambda_max = max(positive_eigenvalues)
        self._lambda_min = min(positive_eigenvalues)

    def _compute_tau_0(self):
        """Вычисление оптимального параметра tau_0."""
        self._tau_0 = 2 / (self._lambda_min + self._lambda_max)

    def _compute_eta(self):
        """Вычисление отношения собственных значений."""
        self._eta = self._lambda_min / self._lambda_max

    def _compute_rho_0(self):
        """Вычисление коэффициента сходимости."""
        self._rho_0 = (1 - self._eta) / (1 + self._eta)

    def _v_k(self, k):
        """Коэффициенты Чебышёва."""
        return np.cos((2 * k - 1) * np.pi / (2 * self._n))

    def _t_k(self, k):
        """Вычисление оптимального шага t_k."""
        return self._tau_0 / (1 + self._rho_0 * self._v_k(k))

    def _compute_x(self, x: np.array):
        """Одна итерация метода Ричардсона."""
        x_k = x
        for k in range(1, self._n + 1):
            t_k = self._t_k(k)
            x_k = (self._b - self._A @ x_k) * t_k + x_k  # Итерационная формула

        return x_k

    def compute(self):
        """Основной метод решения."""
        self._compute_SZ()  # Вычисляем спектр
        self._compute_eta()  # Вычисляем отношение собственных значений
        self._compute_rho_0()  # Вычисляем коэффициент сходимости
        self._compute_tau_0()  # Вычисляем оптимальный параметр tau_0

        x = self._x  # Начальное приближение
        iterations = 0  # Счётчик итераций

        while np.linalg.norm(self._A @ x - self._b) > self._tol:
            x = self._compute_x(x)
            iterations += 1

            if iterations > 1e6:  # Ограничение на максимальное число итераций
                raise ValueError("Метод не сошёлся за разумное число итераций.")

        return x, iterations * self._n


def main():
    # A = np.array([[2, 1, 1],
    #               [1, 2.5, 1],
    #               [1, 1, 3]])
    # print(np.linalg.eigvalsh(A))
    # A = np.array([[-0.168700, 0.353699, 0.008540, 0.733624],
    #               [0.353699, 0.056519, -0.723182, -0.076440],
    #               [0.008540, -0.723182, 0.015938, 0.342333],
    #               [0.733624, -0.076440, 0.342333, -0.045744]])
    # print(np.linalg.eigvalsh(A))
    # A = np.array([[1.00, 0.42, 0.54, 0.66],
    #               [0.42, 1.00, 0.32, 0.44],
    #               [0.54, 0.32, 1.00, 0.22],
    #               [0.66, 0.44, 0.22, 1.00]])
    # print(np.linalg.eigvalsh(A))
    A = np.array([[2, 1],
                 [1, 2]])
    print(np.linalg.eigvalsh(A))
    b = np.array([4,5])
    # b = np.array([1, 1, 1])
    x = np.zeros_like(b)

    richardson = Richardson(A, b, x)

    x, iterations = richardson.compute()
    print(f"Итерационный процесс завершился за {iterations} итераций")
    check_answer(A, b, x)


if __name__ == '__main__':
    main()
