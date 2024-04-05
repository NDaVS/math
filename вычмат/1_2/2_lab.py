import math
from sympy import symbols, diff, log


def function(x):
    return x ** 2 + math.log(x, math.e) - 4


def values(a, b, step):
    table = []
    while a <= b:
        table.append((round(a, 2), function(a)))
        a += step
    table.append((round(a, 2), function(a)))
    return table


def ended_difference(xs):
    table = []
    for i in range(len(xs)):  # заполнение базой для последующего вычисления
        table.append([xs[i], function(xs[i])])

    for i in range(len(xs) - 1):  # отвечает за порядок суммы (столбец)
        for j in range(len(xs) - 1 - i):  # отвечает за выбор нужных ячеек (строка)
            table[j].append((table[j + 1][i + 1] - table[j][i + 1]))
    return table


def print_pretty_table(data, cell_sep='|', header_separator=True):
    table = data
    for i in range(len(table)):
        for j in range(len(table) - i, len(table)):
            table[j].append('--')
    rows = len(data)
    cols = len(data[0])
    table = data
    for i in range(rows):
        for j in range(cols):
            table[i][j] = str(table[i][j])

    new_data = [[' '] + [str(i) for i in range(rows + 1)]]
    for i in range(len(table)):
        new_data += [[f'x_{i}'] + table[i]]

    col_width = []
    rows += 1
    cols += 1
    for col in range(cols):
        columns = [new_data[row][col] for row in range(rows)]
        col_width.append(len(max(columns, key=len)))
    separator = '+'.join('-' * n for n in col_width)

    for i, row in enumerate(range(rows)):
        if i == 1 and header_separator:
            print(separator)

        result = []
        for col in range(cols):
            item = new_data[row][col].rjust(col_width[col])
            result.append(item)
        print(cell_sep.join(result))


def find_interval(x, a, b, step):
    i = 0
    while a + i * step <= b:
        prev = a + i * step
        next = a + step * (i + 1)
        if prev < x < next:
            return i
        i += 1


def t_newton_positive(ed, a, step, x):
    t = (x - a) / step
    result = float(ed[0][1])
    for i in range(1, len(ed[0]) - 1):
        value = float(ed[0][i + 1])
        for j in range(i):
            coef = t - j
            value = value * coef

        value /= math.factorial(i)
        result += value
    return result


def t_newton_negative(ed, b, step, x):
    t = (x - b) / step
    result = float(ed[-1][1])
    for i in range(1, len(ed[0]) - 1):
        value = float(ed[-i - 1][i + 1])
        for j in range(i):
            coef = t + j
            value = value * coef

        value /= math.factorial(i)
        result += value
    return result


def t_gauss_1(ed, x, step, index):
    row_number = index
    t = round((x - float(ed[row_number][0])) / step, 2)
    result = float(ed[row_number][1])
    # row_number -= 1
    for i in range(1, len(ed[0]) - 1):
        value = float(ed[row_number][i + 1])
        coef = 1
        for m in range(i):
            coef *= t - m // 2 - 1 if m % 2 else t + m / 2
        value *= coef

        value /= math.factorial(i)
        result += value
        if i % 2 == 1:
            row_number -= 1

    return result


def t_gauss_2(ed, x, step, index):
    row_number = index+1
    t = round((x - float(ed[row_number][0])) / step, 2)
    result = float(ed[row_number][1])
    row_number -= 1
    for i in range(1, len(ed[0]) - 1):
        
        value = float(ed[row_number][i + 1])
        coef = 1
        for m in range(i):
            coef *= t + m // 2 - 1 if m % 2 else t - m / 2
        value *= coef

        value /= math.factorial(i)
        result += value
        if i % 2 == 0:
            row_number -= 1

    return result


def take_dif(func, x, n):
    new_fun = func
    for _ in range(n):
        new_fun = diff(new_fun, x)
    return new_fun


def omega(a, b, step, x):
    result = 1
    while round(a, 2) <= b:
        result *= (x - a)
        a += step
    return result


def show_answer(x, limits, table, a, b, step, i):
    r_min, r_max = limits
    if i == 0:
        # первый ньютон#
        print(f"{x} - первый ньютон")
        l_n = t_newton_positive(table, a, step, x)
        r1 = abs(l_n - function(x))
        if r_min < r1 < r_max:
            print('nice')
        else:
            print('за работу')

    elif i == 9:
        # второй ньютон#
        print(f"{x} - второй Ньютон")
        l_n = t_newton_negative(table, b, step, x)
        r1 = abs(l_n - function(x))

        if r_min < r1 < r_max:
            print('nice')
        else:
            print('за работу')
    elif a + (i + 1) * step - x < x - a - i * step:
        # второй гаусс#
        print(f"{x} - второй гаусс")
        l_n = t_gauss_2(table, x, step, i)
        r1 = l_n - function(x)
        if r_min < r1 < r_max:
            print('nice')
        else:
            print('за работу')
            print(r1)
            print(function(x))
    else:
        # первый гаусс#
        print(f"{x} - первый гаусс")
        l_n = t_gauss_1(table, x, step, i)
        r1 = l_n - function(x)
        if r_min < r1 < r_max:
            print('nice')
        else:
            print('за работу')


def main():
    a = 1.5
    b = 2
    step = (b - a) / 10
    table = values(a, b, step)
    table = ended_difference([x[0] for x in table])

    print_pretty_table(table)

    X = symbols('x')
    f = X ** 2 + log(X) - 4
    df = take_dif(f, X, 11)
    xs = [1.52, 1.97, 1.77, 1.79]
    for x in xs:
        i = find_interval(x, a, b, step)
        r_min = (df.subs(X, a) / math.factorial(11)) * omega(a, b, step, x)

        r_max = df.subs(X, b) / math.factorial(11) * omega(a, b, step, x)
        limits = [abs(r_min), abs(r_max)]
        limits.sort()
        show_answer(x, limits, table, a, b, step, i)


if __name__ == '__main__':
    main()
