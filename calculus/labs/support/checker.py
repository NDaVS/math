from calculus.labs.support.subs import *
from calculus.labs.support.interpolations import *


def check_results(initial_function, interpolated_function, symbol, start_x, finish_x, points_number, x_value):
    """
         Проверить удволетворяет ли полученная интерполяционная функция необходимой погрешности
         : param initial_function: изначальная функция
         : param interpolated_function: интерполированная функция
         : param symbol: переменная в предоставленных формулах
         : param start_x: начало промежутка интерполирования
         : param finish_x: конец промежутка интерполирования
         : param points_number: количество взятых узлов
         : param x_value: Интересующая точка
    """
    param = -1

    # Инициализация -------------------------------------------------------------------------------
    step = (finish_x - start_x) / points_number
    x_values = [start_x + i * step for i in range(points_number + 1)]
    y_values = [initial_function.subs(symbol, xi) for xi in x_values]
    x_max_value = x_values[y_values.index(max(y_values))]
    x_min_value = x_values[y_values.index(min(y_values))]
    # ---------------------------------------------------------------------------------------------

    R_min = abs((dnf(initial_function, symbol, points_number - param).subs(symbol, x_min_value)
                 / factorial(points_number - param)
                 * omega(start_x, finish_x, points_number, symbol, points_number - param)).subs(symbol, x_value))

    R_max = abs((dnf(initial_function, symbol, points_number - param).subs(symbol, x_max_value)
                 / factorial(points_number - param)
                 * omega(start_x, finish_x, points_number, symbol, points_number - param)).subs(symbol, x_value))

    R_min, R_max = (R_min, R_max) if R_min <= R_max else (R_max, R_min)
    R = abs(interpolated_function.subs(symbol, x_value) - initial_function.subs(symbol, x_value))

    if R_min <= R <= R_max:
        print("Ok")
        print(f"{R_min} <= {R} <= {R_max}")
        return True
    print("Fail")
    print(f"{R_min} <> {R} <> {R_max}")
    return False
