from sympy import *


# ======================================================== лагранжевая интерполяция ==================================
def lagrange(list_of_x, list_of_y, symbol):
    """
         Получить лагранжеву интерполяцию
         : param list_of_x: список значений x
         : param y_values: список значений y
         : param symbol: число для интерполяции
         : return: вернуть результат интерполяции
    """
    ans = 0.0
    for i in range(len(list_of_y)):
        t_ = list_of_y[i]
        for j in range(len(list_of_y)):
            if i != j:
                t_ *= (symbol - list_of_x[j]) / (list_of_x[i] - list_of_x[j])
        ans += t_
    return ans


# ======================================================== Ньютоновская интерполяция =================================
def table(list_of_x, list_of_y):
    """
         Получить таблицу интерполяции Ньютона
         : param x_: значение списка x
         : param y: значение списка y
         : return: вернуть таблицу интерполяции
    """
    quotient = [[0] * len(list_of_x) for _ in range(len(list_of_x))]
    for n_ in range(len(list_of_x)):
        quotient[n_][0] = list_of_y[n_]
    for i in range(1, len(list_of_x)):
        for j in range(i, len(list_of_x)):
            # j-i определяет диагональные элементы
            quotient[j][i] = (quotient[j][i - 1] - quotient[j - 1][i - 1]) / (list_of_x[j] - list_of_x[j - i])
    return quotient


def get_corner(table_of_finite_differences):
    """
         Получить диагональные элементы через таблицу интерполяции
         : param result: результат таблицы интерполяции
         : return: диагональный элемент
    """
    result = []
    for i in range(len(table_of_finite_differences)):
        result.append(table_of_finite_differences[i][i])
    return result


def newton(data_set, search_x, list_of_x):
    """
         Результат интерполяции Ньютона
         : param data_set: диагональ решаемой задачи
         : param search_x: входное значение
         : param list_of_x: исходное значение списка x
         : return: результат интерполяции Ньютона
    """
    result = data_set[0]
    for i in range(1, len(data_set)):
        p = data_set[i]
        for j in range(i):
            p *= (search_x - list_of_x[j])
        result += p
    return result