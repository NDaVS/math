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

def main():
    a = 1.5
    b = 2
    step = (b - a) / 10
    table = values(a - step, b, step)
    table = ended_difference(   [x[0] for x in table])
    for i in range(len(table)):
        for j in range(len(table) - i, len(table)):
            table[j].append('--')
    print_pretty_table(table)


if __name__ == '__main__':
    main()
