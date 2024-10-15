import numpy as np


def find_max_abs_value(A, start_index, end_index):
    row, column, value = 0, 0, 0

    for i in range(start_index, end_index):
        for j in range(start_index, end_index):
            if np.abs(A[i][j]) > np.abs(value):
                row = i
                column = j
                value = A[i][j]
    return row, column, value


def check_permutation(augmented_matrix, column_indices, max_row, max_column, k):
    if max_row != k:
        augmented_matrix[[k, max_row]] = augmented_matrix[[max_row, k]]

    if max_column != k:
        augmented_matrix[:, [max_column, k]] = augmented_matrix[:, [k, max_column]]
        column_indices[[max_column, k]] = column_indices[[k, max_column]]

    return augmented_matrix, column_indices


def answer_permutation(target_vector, order):
    n = len(target_vector)
    x_permuted = np.zeros(n)
    dictionary = dict(zip(order, target_vector))

    for i in range(n):
        x_permuted[i] = dictionary[i]

    return x_permuted


def gauss_with_pivoting(A, b):
    """
    Решение системы линейных уравнений Ax = b методом Гаусса с выбором главного элемента.

    :param A: Коэффициентная матрица (numpy array)
    :param b: Вектор свободных членов (numpy array)
    :return: Решение системы (numpy array)
    """
    n = len(b)
    # Создаем расширенную матрицу [A|b]
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    # Создаём массив последовательности для отслеживаниия изменения порядка столбцов
    column_indices = np.arange(n)

    for k in range(n):
        # Находим наибольший элемент по модулю, а также его координаты
        max_row, max_column, max_abs_value = find_max_abs_value(augmented_matrix, k, n)

        if max_abs_value == 0:
            return -1

        # Переносим его в левый верхний угод подматрицы
        augmented_matrix, column_indices = check_permutation(augmented_matrix, column_indices, max_row, max_column, k)

        # Вычитаем строку с максимальным элементом по модулю, умноженную на коэффициент
        for i in range(k + 1, n):
            augmented_matrix[i] = augmented_matrix[i] + augmented_matrix[k] * \
                                  (- augmented_matrix[i][k] / augmented_matrix[k][k])
        # Нормализуем целевую строку
        augmented_matrix[k] = augmented_matrix[k] / augmented_matrix[k][k]

    # Обратный ход для вычислений
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])) / augmented_matrix[i, i]

    # ставим на свои места нужные элементы
    return answer_permutation(x, column_indices)


if __name__ == '__main__':
    # # # Пример использования
    # A = np.array([[2.1, -4.5, -2.0],
    #               [3, 2.5, 4.3],
    #               [-6, 3.5, 2.5]])
    # b = np.array([19.07, 3.21, -18.25])
    #
    # x = gauss_with_pivoting(A, b)
    # print(x)
    #
    # x = np.linalg.solve(A, b)
    # print(x)

    A = np.array([
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
    ])
    b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])
    
    x = gauss_with_pivoting(A, b)

    if type(x) == int:
        print("Матрица вырожденная")
        exit(0)

    print("Вектор ответа: " + str(x))

    print(
        "Модуль разности нашего решения и решения через библиотеку np: " +
        str(np.linalg.norm(np.linalg.solve(A, b) - x))
    )

    print(
        "Модуль разности произведения матрицы на наш вектор ответа и вектора свободных членов: " +
        str(np.linalg.norm(A @ x - b))
    )

    print(
        "Модуль разности произведения матрицы на  вектор ответа, полученный с помощью библиотеки np, и " +
        "вектора свободных членов: " +
        str(np.linalg.norm(A @ np.linalg.solve(A, b) - b))
    )
