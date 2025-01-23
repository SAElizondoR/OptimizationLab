def factoriales_y_rutas(n):
    """
    Generador que calcula los primeros n factoriales y el valor n! - n - 1.

    :param n: Número de factoriales y rutas por calcular.
    :yield: Tuplas (factorial, rutas) de 1 a n.
    """
    # Casos base
    yield 1, 0
    yield 2, 1
    factorial = 6   # 3! = 6
    for i in range(3, n + 1):
        rutas = factorial - i - 1
        yield factorial, rutas
        factorial *= i + 1

n = 15
print(f"Los primeros {n} factoriales y rutas (n! - n - 1) son:")
for i, (factorial, rutas) in enumerate(factoriales_y_rutas(n), start=1):
    print(f"{i}: Factorial = {factorial}, Rutas = {rutas}")
