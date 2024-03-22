import math
from sympy import symbols, diff, log


def function(x):
    return x ** 2 + math.log(x, math.e) - 4


def values(a, b, step):
    table = []
    while a <= b:
        table.append((a, function(a)))
        a += step
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


def main():
    a = 1.5
    b = 2
    step = (b - a) / 10
    table = values(a, b, step)
    table = ended_difference([x[0] for x in table])
    # print(table)

    print_pretty_table(table)
    xs = [1.52, 1.97, 1.77]
    for x in xs:
        i = find_interval(x, a, b, step)
        if i == 0:
            # первый ньютон#
            print("первый ньютон")
            l_n = t_newton_positive(table, a, step, x)
            print(l_n)
            print(function(x))
        elif i == 9:
            # второй ньютон#
            print("второй ньютон")
            pass
        elif a + (i + 1) * step - x < a + (i) * step - x:
            # второй гаусс#
            print("второй гауус")
            pass
        else:
            # первый гаусс#
            print("первый гауус")
            pass
        print(i)


if __name__ == '__main__':
    main()
