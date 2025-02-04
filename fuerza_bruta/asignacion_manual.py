import numpy as np
from itertools import permutations

def resolver_asignacion(matriz_costos):
    """
    Resuelve el problema de asignación clásico de manera exhaustiva,
    usando todas las permutaciones posibles para asignar trabajos a recursos.

    :param matriz_costos: Matriz cuadrada de costos (n x n) para asignar
    recursos a tareas.
    :return: Tupla con los índices de asignación óptima y el costo total
    mínimo.
    """
    n = matriz_costos.shape[0]
    mejor_costo = float('inf')
    mejor_asignacion = None

    # Generar todas las permutaciones posibles de las asignaciones
    for perm in permutations(range(n)):
        # Calcular el costo total de la asignación actual
        costo_actual = sum(matriz_costos[i, perm[i]] for i in range(n))

        # Actualizar la mejor asignación si se encuentra un costo más bajo
        if costo_actual < mejor_costo:
            mejor_costo = costo_actual
            mejor_asignacion = perm
    
    # Devolver la mejor asignación ajustada y el costo total
    return (np.array(mejor_asignacion) + 1, mejor_costo)

# Ejemplo de matriz de costos (n tareas x n recursos)
costos = np.array([
    [10, 15, 20],
    [25, 30, 35],
    [40, 45, 50]
])

# Resolver el problema de asignación clásico
asignacion, costo_total = resolver_asignacion(costos)

# Mostrar resultados
print("Asignación óptima de trabajos a recursos:")
print(asignacion)
print(f"Costo total mínimo de la asignación: {costo_total:.2f}")
