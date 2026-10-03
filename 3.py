import numpy as np


def check_diagonal_dominance(A):
    """Строгое диагональное преобладание -- достаточное условие сходимости."""
    d = np.abs(np.diag(A))
    off = np.sum(np.abs(A), axis=1) - d
    return bool(np.all(d > off))


def simple_iteration(A, b, eps, max_iter=10_000):
    """Метод простых итераций (Якоби): x^{k+1} = B x^k + c."""
    n = len(b)
    D = np.diag(A)                      # диагональ матрицы A (вектор)
    B = -(A - np.diagflat(D)) / D[:, None]   # B = -D^{-1}(A - D)
    c = b / D                           # c = D^{-1} b

    x = c.copy()                        # начальное приближение x^0 = c
    for k in range(1, max_iter + 1):
        x_new = B @ x + c
        if np.linalg.norm(x_new - x, np.inf) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


def seidel(A, b, eps, max_iter=10_000):
    """Метод Зейделя: новые компоненты используются сразу же."""
    n = len(b)
    x = b / np.diag(A)                  # то же начальное приближение
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = A[i, :i] @ x[:i]           # уже обновлённые x_1..x_{i-1}
            s2 = A[i, i + 1:] @ x_old[i + 1:]  # ещё старые x_{i+1}..x_n
            x[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < eps:
            return x, k
    return x, max_iter


if __name__ == "__main__":
    A = np.array([
        [28.0,  9.0,  -3.0, -7.0],
        [-5.0, 21.0,  -5.0, -3.0],
        [-8.0,  1.0, -16.0,  5.0],
        [ 0.0, -2.0,   5.0,  8.0],
    ])
    b = np.array([-159.0, 63.0, -45.0, 24.0])

    print("Диагональное преобладание:", check_diagonal_dominance(A))
    print("Точное решение (np.linalg.solve):", np.linalg.solve(A, b))
    print()

    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        x_si, k_si = simple_iteration(A, b, eps)
        x_gs, k_gs = seidel(A, b, eps)
        print(f"eps = {eps:g}")
        print(f"  Простые итерации: {k_si:3d} итер., x = {np.round(x_si, 8)}")
        print(f"  Зейдель:          {k_gs:3d} итер., x = {np.round(x_gs, 8)}")