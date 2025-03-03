import numpy as np


def TDMA(a,b,c,f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, b, c, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0]*n

    for i in range(1, n):
        alpha.append(-b[i]/(a[i]*alpha[i-1] + c[i]))
        beta.append((f[i] - a[i]*beta[i-1])/(a[i]*alpha[i-1] + c[i]))

    x[n-1] = beta[n - 1]

    for i in range(n-1, -1, -1):
        x[i - 1] = alpha[i - 1]*x[i] + beta[i - 1]

    return x




def multiply_tridiagonal(a, b, c, x):
    n = len(x)
    result = [0] * n
    for i in range(n):
        if i == 0:
            result[i] = b[i] * x[i] + c[i] * x[i + 1]
        elif i == n - 1:
            result[i] = a[i] * x[i - 1] + b[i] * x[i]
        else:
            result[i] = a[i] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]
    return result


def check_solution(a, b, c, f, x):
    computed_f = multiply_tridiagonal(a, b, c, x)
    return all(abs(cf - ff) < 1 for cf, ff in zip(computed_f, f))


# Входные данные
a = [0, 1, 1, 1]  # Поддиагональ, первый элемент равен 0
b = [4, 4, 4, 4]  # Главная диагональ
c = [1, 1, 1, 0]  # Наддиагональ, последний элемент равен 0
f = [5, 5, 5, 5]  # Вектор правых частей


a = [0, 1, 1, 1]
c = [4, 4, 4, 4]
b = [1, 1, 1, 0]
f = [5, 5, 5, 5]


# /* n - число уравнений (строк матрицы)
# 	 * b - диагональ, лежащая над главной          (нумеруется: [0;n-1], b[n-1]=0)
# 	 * c - главная диагональ матрицы A             (нумеруется: [0;n-1])
# 	 * a - диагональ, лежащая под главной          (нумеруется: [0;n-1], a[0]=0)
# 	 * f - правая часть (столбец)                  (нумеруется: [0;n-1])
# 	 * x - решение, массив x будет содержать ответ (нумеруется: [0;n-1])
# 	 */
# Получаем решение от функции метода монотонной прогонки
x_monotonic = TDMA(a, b, c, f)

# Выводим решение
print("Solution using monotonic preserving tridiagonal method:", x_monotonic)




