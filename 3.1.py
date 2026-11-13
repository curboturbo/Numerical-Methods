"""
Лабораторная 3.3. Метод наименьших квадратов (МНК).
Для табличной функции находим приближающие многочлены
а) 1-й степени и б) 2-й степени через решение нормальной системы МНК,
считаем сумму квадратов ошибок и строим графики.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Исходные данные (вариант 10) ----------
x = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
y = np.array([-1.3734, -1.249, -0.7854, 0.7854, 1.249, 1.3734])


def mnk_polynomial(x, y, m):
    n = m + 1
    A = np.zeros((n, n))   # матрица системы
    b = np.zeros(n)        # правая часть

    for j in range(n):
        for k in range(n):
            A[j, k] = np.sum(x ** (j + k))
        b[j] = np.sum(y * x ** j)

    coeffs = np.linalg.solve(A, b)
    return A, b, coeffs


def poly_value(coeffs, x):
    """Значение многочлена a0 + a1*x + a2*x^2 + ... в точках x."""
    result = np.zeros_like(x, dtype=float)
    for k, a in enumerate(coeffs):
        result += a * x ** k
    return result


def poly_to_string(coeffs):
    """Красивая запись многочлена для вывода."""
    parts = []
    for k, a in enumerate(coeffs):
        if k == 0:
            parts.append(f"{a:.6f}")
        elif k == 1:
            parts.append(f"{a:+.6f}*x")
        else:
            parts.append(f"{a:+.6f}*x^{k}")
    return " ".join(parts)


# ---------- Решение для степеней 1 и 2 ----------
results = {}
for m in (1, 2):
    A, b, coeffs = mnk_polynomial(x, y, m)
    y_approx = poly_value(coeffs, x)
    error_sum = np.sum((y - y_approx) ** 2)   # сумма квадратов ошибок
    results[m] = (coeffs, error_sum)

    print(f"===== Многочлен {m}-й степени =====")
    print("Нормальная система:")
    for j in range(m + 1):
        row = "  ".join(f"{A[j, k]:9.4f}*a{k}" for k in range(m + 1))
        print(f"  {row} = {b[j]:.6f}")
    print("Коэффициенты:", ", ".join(f"a{k} = {c:.6f}" for k, c in enumerate(coeffs)))
    print("P(x) =", poly_to_string(coeffs))
    print(f"Сумма квадратов ошибок: {error_sum:.8f}")
    print()

# ---------- Таблица значений ----------
print("  x_i      y_i      P1(x_i)    P2(x_i)")
for xi, yi in zip(x, y):
    p1 = poly_value(results[1][0], np.array([xi]))[0]
    p2 = poly_value(results[2][0], np.array([xi]))[0]
    print(f"{xi:6.1f} {yi:9.4f} {p1:10.4f} {p2:10.4f}")

# ---------- Графики ----------
x_plot = np.linspace(-5.5, 5.5, 300)

plt.figure(figsize=(9, 6))
plt.scatter(x, y, color="black", zorder=3, label="Табличная функция")
plt.plot(x_plot, poly_value(results[1][0], x_plot), "b-",
         label=f"1-я степень (S = {results[1][1]:.5f})")
plt.plot(x_plot, poly_value(results[2][0], x_plot), "r--",
         label=f"2-я степень (S = {results[2][1]:.5f})")
plt.title("Приближение функции методом наименьших квадратов")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.savefig("mnk_plot.png", dpi=150)
plt.show()