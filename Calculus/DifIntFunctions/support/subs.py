from sympy import diff, symbols


def dnf(func, symbol, n):
    """
         Получить n-ую производную функции func
         : param func: дифференциируемая функция
         : param symbol: переменная по которой ведется дифференциировние
         : param n: количество операций дифференциировния
    """
    new_fun = func
    for _ in range(n):
        new_fun = diff(new_fun, symbol)
    return new_fun


def omega(start_value, finish_value, points_number, symbol, counter=0):
    """
         Получить множитель омега для вычисления погрешности
         : param start_value: начало промежутка интерполирования
         : param finish_value: конец промежутка интерполирования
         : param points_number: количество равноудаленных узлов сетки
         : param symbol: переменная, от которой зависит омега
         : param counter: количество взятых узлов
    """
    counter = counter if bool(counter) else points_number

    step = (finish_value - start_value) / points_number
    result = 1
    k = 0
    while start_value <= finish_value and k < counter:
        result *= (symbol - start_value)
        start_value += step
        k += 1
    return result