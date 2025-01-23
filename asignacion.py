"""
Módulo que resuelve el problema de asignación de recursos usando el algoritmo
Húngaro.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

def resolver_problema_asignacion(matriz_costos_local):
    """
    Resuelve el problema de asignación usando el algoritmo Húngaro
    (método de asignación lineal).
    
    :param matriz_costos_local: Matriz cuadrada de costos (n x n) para asignar
    recursos a tareas.
    :return: Tupla con los índices de asignación óptima ajustados y el costo
    total.
    """
    # Aplicar el algoritmo Húngaro para encontrar la asignación óptima
    fila_asignadas, columna_asignadas = \
        linear_sum_assignment(matriz_costos_local)

    # Devolver los índices ajustados y el costo total de la asignación
    return (fila_asignadas, columna_asignadas,
        matriz_costos_local[fila_asignadas, columna_asignadas].sum())

# Matriz de tiempos para cada nadador (filas) y estilo (columnas)
tiempos = np.array([
    [37.7, 43.4, 33.3, 29.2],  # Carlos
    [32.9, 33.1, 28.5, 26.4],  # Cristy
    [33.8, 42.2, 38.9, 29.6],  # David
    [37.0, 34.7, 30.4, 28.5],  # Antony
    [35.4, 41.8, 33.6, 31.1]   # José
])

# Dado que hay 5 nadadores y 4 estilos, necesitamos agregar una columna
# ficticia con tiempos suficientemente grandes para representar un estilo que
# no se asignará.
tiempos_completos = np.hstack([tiempos, np.full((5, 1), 1e6)])

# Resolver el problema de asignación
filas, columnas, costo_total = resolver_problema_asignacion(tiempos_completos)

# Asignaciones de los nadadores a los estilos
nadadores = ['Carlos', 'Cristy', 'David', 'Antony', 'José']
estilos = ['Dorso', 'Pecho', 'Mariposa', 'Libre']

# Mostrar los resultados
nadadores = ['Carlos', 'Cristy', 'David', 'Antony', 'José']
estilos = ['Dorso', 'Pecho', 'Mariposa', 'Libre']

# Mostrar las asignaciones de los nadadores a los estilos
print("Asignaciones óptimas (nadador -> estilo):")
for nadador_idx, estilo_idx in zip(filas, columnas):
    if estilo_idx < 4:  # Excluir el estilo ficticio (índice 4)
        print(f"{nadadores[nadador_idx]} -> {estilos[estilo_idx]}")

print(f"Costo total (tiempo total de la asignación): "
      f"{costo_total - 1e6:.2f} segundos")