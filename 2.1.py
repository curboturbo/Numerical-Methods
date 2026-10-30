import math
import numpy as np
import matplotlib.pyplot as plt



def f(x):
    return math.sin(x) - 2 * x ** 2 + 0.5


def df(x):
    return math.cos(x) - 4 * x


# Для простой итерации приводим f(x)=0 к виду x = phi(x):
#   2x^2 = sin x + 0.5  =>  x = sqrt((sin x + 0.5) / 2)   (положительный корень)
def phi(x):
    return math.sqrt((math.sin(x) + 0.5) / 2)


def dphi(x):
    return math.cos(x) / (4 * phi(x))


def simple_iteration(x0, eps, max_iter=10000):
    """Возвращает (корень, число итераций, список приближений)."""
    xs = [x0]
    x = x0
    for k in range(1, max_iter + 1):
        x_new = phi(x)
        xs.append(x_new)
        if abs(x_new - x) < eps:
            return x_new, k, xs
        x = x_new
    return x, max_iter, xs


def newton(x0, eps, max_iter=10000):
    xs = [x0]
    x = x0
    for k in range(1, max_iter + 1):
        d = df(x)
        if d == 0:
            raise ZeroDivisionError("Производная равна нулю")
        x_new = x - f(x) / d
        xs.append(x_new)
        if abs(x_new - x) < eps:
            return x_new, k, xs
        x = x_new
    return x, max_iter, xs


def bisection(a, b, eps, max_iter=200):
    rows = []
    k = 0
    while k < max_iter:
        c = (a + b) / 2
        fa, fb, fc = f(a), f(b), f(c)
        rows.append((k, a, b, fa, fb, c, fc))
        if (b - a) / 2 < eps or fc == 0:
            break
        if fa * fc < 0:
            b = c
        else:
            a = c
        k += 1
    return c, rows


BISECTION_HEADER = ["k", "a(k)", "b(k)", "f(a(k))", "f(b(k))",
                    "(a(k)+b(k))/2", "f((a(k)+b(k))/2)"]


def print_bisection_table(rows):
    print(f"{'k':>3} | {'a(k)':>12} | {'b(k)':>12} | {'f(a(k))':>12} | "
          f"{'f(b(k))':>12} | {'(a+b)/2':>12} | {'f((a+b)/2)':>12}")
    print("-" * 100)
    for k, a, b, fa, fb, c, fc in rows:
        print(f"{k:3d} | {a:12.8f} | {b:12.8f} | {fa:12.3e} | "
              f"{fb:12.3e} | {c:12.8f} | {fc:12.3e}")


def read_eps(default=1e-6):
    while True:
        s = input(f"Введите точность eps (например 1e-4 или 0.0001) "
                  f"[по умолчанию {default:g}]: ").strip().replace(",", ".")
        if not s:
            return default
        try:
            eps = float(s)
            if eps <= 0:
                raise ValueError
            return eps
        except ValueError:
            print("Некорректное значение, попробуйте ещё раз.")


def plot_function(a=-1.0, b=1.5, filename="graph_f.png"):
    xs = np.linspace(a, b, 500)
    ys = np.sin(xs) - 2 * xs ** 2 + 0.5
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, label=r"$f(x)=\sin x-2x^2+0.5$")
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(0, color="k", lw=0.8)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("График функции (выбор начального приближения)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def find_bracket(a=0.0, b=2.0, step=0.1):
    x = a
    while x < b:
        if f(x) * f(x + step) < 0:
            return x, x + step
        x += step
    return None


def main():
    plot_function()

    bracket = find_bracket()
    a, b = bracket
    x0 = (a + b) / 2
    print(f"Отрезок смены знака: [{a:.2f}, {b:.2f}], x0 = {x0:.3f}")

    # нахожднеи q для метода итераций
    q = max(abs(dphi(t)) for t in np.linspace(a, b, 100))
    print(f"max|phi'(x)| на отрезке = {q:.4f}  (<1 -> метод сходится)\n")

    # Эталонный корень (очень высокая точность)
    root_ref, _, _ = newton(x0, 1e-15)
    print(f"Эталонный корень: {root_ref:.15f}\n")

    #Решение для набора точностей
    print(f"{'eps':>10} | {'Прост. итер.: x':>18} {'N':>4} | {'Ньютон: x':>18} {'N':>3}")
    print("-" * 66)
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]:
        xi, ni, _ = simple_iteration(x0, eps)
        xn, nn, _ = newton(x0, eps)
        print(f"{eps:10.0e} | {xi:18.12f} {ni:4d} | {xn:18.12f} {nn:3d}")

    #Зависимость погрешности от числа итераций
    _, _, xs_si = simple_iteration(x0, 1e-14)
    _, _, xs_nw = newton(x0, 1e-14)
    err_si = [abs(x - root_ref) for x in xs_si]
    err_nw = [abs(x - root_ref) for x in xs_nw]

    print("\nПогрешность |x_k - x*| по итерациям")
    print(f"{'k':>3} | {'Прост. итерация':>18} | {'Ньютон':>14}")
    for k in range(max(len(err_si), len(err_nw))):
        s = f"{err_si[k]:18.3e}" if k < len(err_si) else " " * 18
        n = f"{err_nw[k]:14.3e}" if k < len(err_nw) else ""
        print(f"{k:3d} | {s} | {n}")

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(range(len(err_si)), [max(e, 1e-17) for e in err_si], "o-",
                 label="Простая итерация")
    plt.semilogy(range(len(err_nw)), [max(e, 1e-17) for e in err_nw], "s-",
                 label="Метод Ньютона")
    plt.grid(True, which="both")
    plt.xlabel("Номер итерации k")
    plt.ylabel(r"Погрешность $|x_k - x^*|$")
    plt.title("Зависимость погрешности от числа итераций")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_vs_iter.png", dpi=150)
    plt.close()

    # ---------- Метод половинного деления: таблица по шагам ----------
    eps = read_eps()
    root_bis, rows = bisection(a, b, eps)
    print(f"\nМетод половинного деления на [{a:.2f}, {b:.2f}], eps = {eps:g}")
    print_bisection_table(rows)
    print(f"\nКорень: {root_bis:.10f}, число итераций: {len(rows)}, "
          f"погрешность |x - x*| = {abs(root_bis - root_ref):.3e}")

    # Для сравнения — остальные методы с той же точностью
    xi, ni, _ = simple_iteration(x0, eps)
    xn, nn, _ = newton(x0, eps)
    print(f"\nПри eps = {eps:g}:")
    print(f"  Половинное деление: x = {root_bis:.10f}, итераций: {len(rows)}")
    print(f"  Простая итерация:   x = {xi:.10f}, итераций: {ni}")
    print(f"  Метод Ньютона:      x = {xn:.10f}, итераций: {nn}")


if __name__ == "__main__":
    main()