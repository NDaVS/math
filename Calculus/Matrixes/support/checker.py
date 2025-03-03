import numpy as np


def check_answer(A, b, x):
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
