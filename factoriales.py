"""
Módulo para calcular factoriales y rutas del problema del agente viajero.
Contiene funciones para generar los primeros n factoriales y calcular 
el valor de n! - n - 1.
"""
def factoriales_y_rutas(n):
    """
    Generador que calcula los primeros n factoriales y el valor n! - n - 1.

    :param n: Número de factoriales y rutas por calcular.
    :yield: Tuplas (factorial, rutas) de 1 a n.
    """
    # Casos base
    yield 1, 0
    yield 2, 1
    fact = 6   # 3! = 6
    for j in range(3, n + 1):
        rutas_calculadas = fact - j - 1
        yield fact, rutas_calculadas
        fact *= j + 1

N_VALUE = 15
print(f"Los primeros {N_VALUE} factoriales y rutas (n! - n - 1) son:")
for index, (factorial, rutas) in enumerate(factoriales_y_rutas(N_VALUE), start=1):
    print(f"{index}: Factorial = {factorial}, Rutas = {rutas}")
