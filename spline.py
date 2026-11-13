"""
Лабораторная 3.2. Кубический сплайн с нулевой кривизной на концах
(естественный сплайн): S''(x0) = S''(x4) = 0.
Вычисляем значение сплайна в точке X* = -0.5 и строим график.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Исходные данные (вариант 10) ----------
x = np.array([-3.0, -1.0, 1.0, 3.0, 5.0])
f = np.array([-1.2490, -0.78540, 0.78540, 1.2490, 1.3734])
X_star = -0.5

n = len(x) - 1                        # число отрезков (4)
h = [x[i + 1] - x[i] for i in range(n)]   # шаги h_i = x_{i+1} - x_i


def solve_tridiagonal(a, b, c, d):
    """Метод прогонки. a - нижняя, b - главная, c - верхняя диагонали, d - правая часть."""
    m = len(d)
    P = [0.0] * m
    Q = [0.0] * m
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, m):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denom if i < m - 1 else 0.0
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom
    res = [0.0] * m
    res[-1] = Q[-1]
    for i in range(m - 2, -1, -1):
        res[i] = P[i] * res[i + 1] + Q[i]
    return res


# ---------- Коэффициенты c_i ----------
# Сплайн на отрезке [x_{i-1}, x_i]:
#   S_i(x) = a_i + b_i (x - x_{i-1}) + c_i (x - x_{i-1})^2 + d_i (x - x_{i-1})^3
# Нулевая кривизна на концах: c_1 = 0 и c_{n+1} = 0.
# Для внутренних узлов i = 2..n:
#   h_{i-1} c_{i-1} + 2 (h_{i-1} + h_i) c_i + h_i c_{i+1} = 3 ( (f_i - f_{i-1})/h_i - (f_{i-1} - f_{i-2})/h_{i-1} )
# Ниже индексы с нуля: c[0] = 0, c[n] = 0, неизвестны c[1..n-1].

# Система для внутренних c[1], c[2], c[3]
size = n - 1
A_low = [0.0] * size
A_mid = [0.0] * size
A_up = [0.0] * size
rhs = [0.0] * size

for k in range(size):
    i = k + 1   # номер узла
    A_low[k] = h[i - 1]
    A_mid[k] = 2 * (h[i - 1] + h[i])
    A_up[k] = h[i]
    rhs[k] = 3 * ((f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1])

c_inner = solve_tridiagonal(A_low, A_mid, A_up, rhs)
c = [0.0] + c_inner + [0.0]       # c[0] = c[n] = 0


a = [f[i] for i in range(n)]
b = [0.0] * n
d = [0.0] * n
for i in range(n):
    b[i] = (f[i + 1] - f[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
    d[i] = (c[i + 1] - c[i]) / (3 * h[i])


def spline(t):
    i = 0
    while i < n - 1 and t > x[i + 1]:
        i += 1
    dx = t - x[i]
    return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3


print("Коэффициенты сплайна на отрезках S_i(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3")
for i in range(n):
    print(f"[{x[i]:5.1f}; {x[i+1]:5.1f}]  a={a[i]:9.5f}  b={b[i]:9.5f}  c={c[i]:9.5f}  d={d[i]:9.5f}")

result = spline(X_star)
print(f"\nS({X_star}) = {result:.6f}")

print("\nПроверка в узлах:")
for xi, fi in zip(x, f):
    print(f"x={xi:5.1f}  f={fi:8.4f}  S={spline(xi):8.4f}")


xs = np.linspace(x[0], x[-1], 400)
ys = [spline(t) for t in xs]

margin = 3.0
xs_left = np.linspace(x[0] - margin, x[0], 100)
xs_right = np.linspace(x[-1], x[-1] + margin, 100)
ys_left = [spline(t) for t in xs_left]
ys_right = [spline(t) for t in xs_right]

plt.figure(figsize=(10, 6))
plt.plot(xs, ys, "b-", linewidth=2, label="Кубический сплайн (на интервале)")
plt.plot(xs_left, ys_left, "--", color="gold", linewidth=2, label="Продолжение за пределы интервала")
plt.plot(xs_right, ys_right, "--", color="gold", linewidth=2)
plt.scatter(x, f, color="black", zorder=3, label="Узлы интерполяции")
plt.scatter([X_star], [result], color="red", zorder=4, label=f"S({X_star}) = {result:.4f}")
plt.title("Кубический сплайн с нулевой кривизной на концах")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.savefig("spline_plot.png", dpi=150)
plt.show()