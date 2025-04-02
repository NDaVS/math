import sympy as sp
import matplotlib.pyplot as plt

# Определяем переменные
x = sp.symbols('x')
n = sp.symbols('n', integer=True, positive=True)
l = sp.symbols('l', positive=True)
# Определяем функцию f(x) = x^2 и границы
f_x = -x
base = sp.sin(2*n*x)
a, b = 0, sp.pi /2

# Формула для коэффициентов Фурье
A_n = sp.integrate(f_x * base, (x, a, b)) / sp.integrate(base**2, (x, a, b))

# Вывод общего выражения для коэффициентов в LaTeX
latex_code = sp.latex(sp.simplify(A_n))
print(latex_code)


if latex_code[:13] == r'\begin{cases}':
    latex_code = latex_code[14:-12]
    latex_code = latex_code.replace('&', '')
    latex_code = latex_code.replace(r'\\', ',')

print(latex_code)

plt.figure(figsize=(6, 2))
plt.text(0.5, 0.5, f"${latex_code}$", fontsize=20, ha='center', va='center')
plt.axis("off")  # Отключаем оси
plt.show()
