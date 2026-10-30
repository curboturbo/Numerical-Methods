import numpy as np
import matplotlib.pyplot as plt

A = 1.0


# ---------- Система F(x) = 0 и матрица Якоби ----------
def F(x):
    x1, x2 = x
    return np.array([x1 - np.cos(x2) - A,
                     x2 - np.sin(x1) - A])


def J(x):
    x1, x2 = x
    return np.array([[1.0, np.sin(x2)],
                     [-np.cos(x1), 1.0]])


# ---------- Приведение к виду x = phi(x) для простой итерации ----------
#   x1 = a + cos(x2)
#   x2 = a + sin(x1)
def phi(x):
    x1, x2 = x
    return np.array([A + np.cos(x2),
                     A + np.sin(x1)])


def J_phi(x):
    x1, x2 = x
    return np.array([[0.0, -np.sin(x2)],
                     [np.cos(x1), 0.0]])


def norm(v):
    """Норма-максимум (бесконечная норма) вектора."""
    return np.max(np.abs(v))


def matrix_norm(M):
    """Норма матрицы, согласованная с нормой-максимумом (макс. сумма по строкам)."""
    return np.max(np.sum(np.abs(M), axis=1))


# ---------- Методы ----------
def simple_iteration(x0, eps, max_iter=10000):
    """Возвращает (решение, число итераций, список приближений)."""
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    for k in range(1, max_iter + 1):
        x_new = phi(x)
        xs.append(x_new.copy())
        if norm(x_new - x) < eps:
            return x_new, k, xs
        x = x_new
    return x, max_iter, xs


def newton(x0, eps, max_iter=1000):
    """Метод Ньютона: на каждом шаге решается линейная система J(x)*dx = -F(x)."""
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    for k in range(1, max_iter + 1):
        dx = np.linalg.solve(J(x), -F(x))
        x_new = x + dx
        xs.append(x_new.copy())
        if norm(x_new - x) < eps:
            return x_new, k, xs
        x = x_new
    return x, max_iter, xs




def graphical_guess(lo=0.0, hi=3.0, step=0.05):
    """Численный аналог графического метода: точка сетки в положительной
    области, где обе кривые проходят ближе всего друг к другу."""
    best, best_val = None, np.inf
    for x1 in np.arange(lo, hi + step, step):
        for x2 in np.arange(lo, hi + step, step):
            val = norm(F((x1, x2)))
            if val < best_val:
                best, best_val = (x1, x2), val
    return np.round(best, 2)


def find_all_solutions(lo=-3.0, hi=5.0, n=17):
    """Ищет все различные решения методом Ньютона из сетки стартовых точек."""
    found = []
    for s1 in np.linspace(lo, hi, n):
        for s2 in np.linspace(lo, hi, n):
            try:
                sol, _, _ = newton((s1, s2), 1e-12, max_iter=60)
            except np.linalg.LinAlgError:
                continue
            if norm(F(sol)) < 1e-9 and not any(norm(sol - s) < 1e-6 for s in found):
                found.append(sol)
    return found


def read_eps(default=1e-6):
    """Запрашивает точность eps у пользователя (Enter = значение по умолчанию)."""
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


def main():
    # --- 1. Все решения системы и выбор положительного ---
    sols = find_all_solutions()
    print("Найденные решения системы:")
    for s in sols:
        mark = "  <- оба неизвестных положительны" if np.all(s > 0) else ""
        print(f"  x1 = {s[0]:.10f}, x2 = {s[1]:.10f}{mark}")
    positive = [s for s in sols if np.all(s > 0)]
    if not positive:
        print("Положительных решений не найдено")
        return

    # --- 2. Начальное приближение по графику ---
    x0 = graphical_guess()
    print(f"\nНачальное приближение (по графику): x0 = ({x0[0]:.2f}; {x0[1]:.2f})")

    # --- 3. Эталонное решение и проверка сходимости простой итерации ---
    ref, _, _ = newton(x0, 1e-15)
    q = matrix_norm(J_phi(ref))
    print(f"Эталонное решение: x1 = {ref[0]:.15f}, x2 = {ref[1]:.15f}")
    print(f"||phi'(x*)|| = {q:.4f}  (<1 -> простая итерация сходится)\n")

    # --- 4. Число итераций для разных точностей ---
    print(f"{'eps':>8} | {'Прост. итерация: x1':>20} {'x2':>14} {'N':>4} | "
          f"{'Ньютон: x1':>14} {'x2':>14} {'N':>3}")
    print("-" * 94)
    for e in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]:
        xi, ni, _ = simple_iteration(x0, e)
        xn, nn, _ = newton(x0, e)
        print(f"{e:8.0e} | {xi[0]:20.12f} {xi[1]:14.12f} {ni:4d} | "
              f"{xn[0]:14.12f} {xn[1]:14.12f} {nn:3d}")

    # --- 5. Зависимость погрешности от числа итераций ---
    _, _, xs_si = simple_iteration(x0, 1e-14)
    _, _, xs_nw = newton(x0, 1e-14)
    err_si = [norm(x - ref) for x in xs_si]
    err_nw = [norm(x - ref) for x in xs_nw]

    print("\nПогрешность ||x_k - x*|| по итерациям")
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
    plt.ylabel(r"Погрешность $\|x_k - x^*\|_\infty$")
    plt.title("Зависимость погрешности от числа итераций")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_vs_iter_system.png", dpi=150)
    plt.close()

    # --- 6. Пошаговые таблицы для введённой точности ---
    eps = read_eps()
    for name, method in [("Простая итерация", simple_iteration),
                         ("Метод Ньютона", newton)]:
        sol, n, xs = method(x0, eps)
        print(f"\n{name}, eps = {eps:g}")
        print(f"{'k':>3} | {'x1(k)':>14} | {'x2(k)':>14} | {'||x(k)-x(k-1)||':>16}")
        print("-" * 58)
        for k, x in enumerate(xs):
            d = f"{norm(x - xs[k - 1]):16.3e}" if k > 0 else " " * 16
            print(f"{k:3d} | {x[0]:14.10f} | {x[1]:14.10f} | {d}")
        print(f"Решение: x1 = {sol[0]:.10f}, x2 = {sol[1]:.10f}, итераций: {n}, "
              f"погрешность ||x - x*|| = {norm(sol - ref):.3e}")


if __name__ == "__main__":
    main()