def progonka(a, b, c, d):
    """
    Метод прогонки для трёхдиагональной СЛАУ.
    a[i] - элемент под главной диагональю (a[0] не используется)
    b[i] - элемент главной диагонали
    c[i] - элемент над главной диагональю (c[n-1] не используется)
    d[i] - правая часть
    """
    n = len(b)
    p = [0.0] * n   # прогоночные коэффициенты alpha
    q = [0.0] * n   # прогоночные коэффициенты beta

    # Прямой ход
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * p[i - 1]
        if abs(denom) < 1e-12:
            raise ZeroDivisionError("Нулевой знаменатель: метод прогонки неприменим")
        p[i] = -c[i] / denom if i < n - 1 else 0.0
        q[i] = (d[i] - a[i] * q[i - 1]) / denom

    # Обратный ход
    x = [0.0] * n
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x


def read_input():
    n = int(input("Размерность системы n: "))
    a = [0.0] + list(map(float, input(f"Нижняя диагональ ({n-1} чисел): ").split()))
    b = list(map(float, input(f"Главная диагональ ({n} чисел): ").split()))
    c = list(map(float, input(f"Верхняя диагональ ({n-1} чисел): ").split())) + [0.0]
    d = list(map(float, input(f"Правая часть ({n} чисел): ").split()))
    return a, b, c, d


if __name__ == "__main__":
    # Данные вашей системы (можно заменить на read_input())
    a = [0, 6, -3, -9, -5]       # нижняя диагональ
    b = [-7, 12, 5, 21, -6]      # главная диагональ
    c = [-6, 0, 0, 8, 0]         # верхняя диагональ
    d = [-75, 126, 13, -40, -24] # правая часть

    # a, b, c, d = read_input()  # для ручного ввода

    x = progonka(a, b, c, d)
    for i, xi in enumerate(x, 1):
        print(f"x{i} = {xi:.6f}")

    # Проверка: невязка A*x - d
    n = len(b)
    res = []
    for i in range(n):
        s = b[i] * x[i]
        if i > 0:
            s += a[i] * x[i - 1]
        if i < n - 1:
            s += c[i] * x[i + 1]
        res.append(abs(s - d[i]))
    print("Максимальная невязка:", max(res))