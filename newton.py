"""
Лабораторная 3.1. Интерполяционные многочлены Лагранжа и Ньютона.
Функция y = arctg(x), X* = -0.5.
  а) узлы X = -3, -1, 1, 3
  б) узлы X = -3, 0, 1, 3
Для каждого набора узлов строим многочлены Лагранжа и Ньютона,
считаем значение в X* и погрешность интерполяции.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import atan, factorial

X_star = -0.5


def f(x):
    return np.arctan(x)


def f4(x):
    """4-я производная arctg(x) (нужна для теоретической оценки погрешности)."""
    return -24 * x * (x ** 2 - 1) / (1 + x ** 2) ** 4


# ---------- Многочлен Лагранжа ----------
def lagrange(xs, ys, t):
    """L(t) = sum_i y_i * prod_{j != i} (t - x_j) / (x_i - x_j)."""
    total = 0.0
    for i in range(len(xs)):
        term = ys[i]
        for j in range(len(xs)):
            if j != i:
                term *= (t - xs[j]) / (xs[i] - xs[j])
        total += term
    return total


# ---------- Многочлен Ньютона ----------
def divided_differences(xs, ys):
    """Возвращает коэффициенты Ньютона f[x0], f[x0,x1], f[x0,x1,x2], ..."""
    n = len(xs)
    table = [list(ys)]                      # первый столбец - сами значения
    for k in range(1, n):
        prev = table[-1]
        col = [(prev[i + 1] - prev[i]) / (xs[i + k] - xs[i]) for i in range(n - k)]
        table.append(col)
    return table


def newton(xs, coeffs, t):
    """N(t) = c0 + c1 (t-x0) + c2 (t-x0)(t-x1) + ..."""
    result = coeffs[0]
    product = 1.0
    for k in range(1, len(coeffs)):
        product *= (t - xs[k - 1])
        result += coeffs[k] * product
    return result


def solve(name, xs):
    xs = list(xs)
    ys = [float(f(v)) for v in xs]

    table = divided_differences(xs, ys)
    coeffs = [col[0] for col in table]      # верхняя диагональ таблицы

    L = lagrange(xs, ys, X_star)
    N = newton(xs, coeffs, X_star)
    exact = float(f(X_star))

    # фактическая погрешность
    err = abs(exact - L)

    # теоретическая оценка: |R| <= max|f''''| / 4! * |prod (X* - x_i)|
    grid = np.linspace(min(xs), max(xs), 2001)
    M4 = np.max(np.abs(f4(grid)))
    omega = np.prod([X_star - v for v in xs])
    bound = M4 / factorial(len(xs)) * abs(omega)

    print(f"===== {name}: узлы {xs} =====")
    print("y_i =", [round(v, 6) for v in ys])
    print("Разделённые разности (коэффициенты Ньютона):",
          [round(c, 6) for c in coeffs])
    print(f"L({X_star}) = {L:.6f}")
    print(f"N({X_star}) = {N:.6f}")
    print(f"f({X_star}) = arctg({X_star}) = {exact:.6f}")
    print(f"Фактическая погрешность |f(X*) - P(X*)| = {err:.6f}")
    print(f"Теоретическая оценка погрешности       <= {bound:.6f}")
    print()
    return xs, ys, L


# ---------- Решение ----------
res_a = solve("а)", [-3, -1, 1, 3])
res_b = solve("б)", [-3, 0, 1, 3])

# ---------- График ----------
t = np.linspace(-3.5, 3.5, 400)
plt.figure(figsize=(10, 6))
plt.plot(t, f(t), "k-", label="y = arctg(x)")
# Кривые а) и б) совпадают, поэтому рисуем б) толстой красной линией,
# а поверх неё а) тонкой синей пунктирной - так видны обе.
xs, ys, L = res_b
plt.plot(t, [lagrange(xs, ys, v) for v in t], "r-", linewidth=5, alpha=0.5,
         label="Интерполяция б) (узлы -3, 0, 1, 3)")
plt.scatter(xs, ys, color="red", s=60, zorder=3)

xs, ys, L = res_a
plt.plot(t, [lagrange(xs, ys, v) for v in t], "b--", linewidth=2, zorder=3,
         label="Интерполяция а) (узлы -3, -1, 1, 3)")
plt.scatter(xs, ys, color="blue", s=25, zorder=4)
plt.scatter([X_star], [f(X_star)], color="green", zorder=4, label="X* = -0.5")
plt.title("Интерполяция arctg(x) многочленом 3-й степени")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.savefig("interp_plot.png", dpi=150)
plt.show()