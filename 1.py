"""
LU-разложение с выбором главного элемента (по столбцу): P·A = L·U.
Решение СЛАУ, вычисление определителя и обратной матрицы.
Без внешних библиотек (numpy используется только для самопроверки, если установлен).
"""


def lu_decompose(A):
    """
    Возвращает (L, U, P, sign), где:
      P — список перестановок строк (P[i] — номер исходной строки, ставшей i-й),
      L — нижняя треугольная с единицами на диагонали,
      U — верхняя треугольная,
      sign — знак перестановки (+1 / -1) для определителя.
    """
    n = len(A)
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    P = list(range(n))
    sign = 1

    for k in range(n):
        # выбор главного элемента в столбце k
        pivot = max(range(k, n), key=lambda i: abs(U[i][k]))
        if abs(U[pivot][k]) < 1e-12:
            raise ValueError("Матрица вырождена (нулевой главный элемент)")

        if pivot != k:
            U[k], U[pivot] = U[pivot], U[k]
            L[k], L[pivot] = L[pivot], L[k]  # уже вычисленные множители тоже переставляем
            P[k], P[pivot] = P[pivot], P[k]
            sign = -sign

        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]
            L[i][k] = m
            for j in range(k, n):
                U[i][j] -= m * U[k][j]

    for i in range(n):
        L[i][i] = 1.0
    return L, U, P, sign


def solve_lu(L, U, P, b):
    """Решение A·x = b по готовому разложению P·A = L·U."""
    n = len(L)
    pb = [b[P[i]] for i in range(n)]

    # L·y = P·b — прямая подстановка
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(L[i][j] * y[j] for j in range(i))

    # U·x = y — обратная подстановка
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def determinant(U, sign):
    """det A = sign · произведение диагонали U."""
    det = float(sign)
    for i in range(len(U)):
        det *= U[i][i]
    return det


def inverse(L, U, P):
    """Обратная матрица: решаем A·x = e_j для каждого столбца единичной матрицы."""
    n = len(L)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve_lu(L, U, P, e))
    # cols[j] — j-й столбец обратной матрицы
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def mat_mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def print_matrix(M, title="", fmt="{:12.6f}"):
    if title:
        print(title)
    for row in M:
        print(" ".join(fmt.format(v) for v in row))
    print()


if __name__ == "__main__":
    A = [
        [-7.0,  3.0, -4.0,  7.0],
        [ 8.0, -1.0, -7.0,  6.0],
        [ 9.0,  9.0,  3.0, -6.0],
        [-7.0, -9.0, -8.0, -5.0],
    ]
    b = [-126.0, 29.0, 27.0, 34.0]

    L, U, P, sign = lu_decompose(A)

    print_matrix(L, "Матрица L:")
    print_matrix(U, "Матрица U:")
    print("Перестановка строк P:", P, "\n")

    # проверка P·A = L·U
    PA = [A[P[i]] for i in range(len(A))]
    LU = mat_mul(L, U)
    err = max(abs(PA[i][j] - LU[i][j]) for i in range(4) for j in range(4))
    print(f"Проверка: max|P·A - L·U| = {err:.2e}\n")

    x = solve_lu(L, U, P, b)
    print("Решение СЛАУ:")
    for i, v in enumerate(x, 1):
        print(f"  x{i} = {v:.6f}")
    print()

    # невязка A·x - b
    res = max(abs(sum(A[i][j] * x[j] for j in range(4)) - b[i]) for i in range(4))
    print(f"Невязка max|A·x - b| = {res:.2e}\n")

    det = determinant(U, sign)
    print(f"Определитель det A = {det:.6f}\n")

    A_inv = inverse(L, U, P)
    print_matrix(A_inv, "Обратная матрица A^(-1):")

    E = mat_mul(A, A_inv)
    err_inv = max(abs(E[i][j] - (1.0 if i == j else 0.0)) for i in range(4) for j in range(4))
    print(f"Проверка: max|A·A^(-1) - E| = {err_inv:.2e}")

    # сверка с numpy (необязательно)
    try:
        import numpy as np
        print("\nСверка с numpy:")
        print("  x   =", np.linalg.solve(A, b))
        print("  det =", np.linalg.det(A))
    except ImportError:
        pass