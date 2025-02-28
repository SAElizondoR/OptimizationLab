from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

def generar_solucion_voraz(
        articulos: np.ndarray,
        capacidad_mochila: int
) -> Tuple[np.ndarray, int, int]:
    """
    Generar una solución usando un enfoque voraz basado en la relación
    beneficio/peso.

    Args:
        articulos (np.ndarray): Vector de artículos disponibles
        capacidad_mochila (int): Capacidad máxima de la mochila
    
    Returns:
        Tuple: (mascara_seleccion, peso_total, beneficio_total)
    """
    pesos = articulos['peso']
    beneficios = articulos['beneficio']

    # Calcular la relación beneficio/peso y ordenar descendentemente
    relacion_beneficio_peso = beneficios / pesos
    orden_descendente = np.argsort(-relacion_beneficio_peso)

    # Seleccionar artículos hasta llenar la capacidad
    pesos_ordenados = pesos[orden_descendente]
    pesos_acumulados = pesos_ordenados.cumsum()
    maximo_seleccionable = np.searchsorted(
        pesos_acumulados,
        capacidad_mochila,
        side='right'
    )

    # Crear máscara de selección
    mascara_seleccion = np.zeros_like(orden_descendente, dtype=bool)
    mascara_seleccion[orden_descendente[:maximo_seleccionable]] = True

    # Calcular métricas finales
    peso_total = (pesos_acumulados[maximo_seleccionable-1]
                  if maximo_seleccionable > 0 else 0)
    beneficio_total = beneficios[
        orden_descendente[:maximo_seleccionable]
    ].sum()

    return mascara_seleccion, peso_total, beneficio_total

def optimizar_solucion(
        articulos: np.ndarray,
        capacidad_mochila: int,
        mascara_seleccion: np.ndarray,
        peso_actual: int,
        beneficio_actual: int,
        max_repeticiones: int,
        umbral_mejora: int,
        verbose: bool
    ) -> Dict[str, float | List[int] | int]:
    """
    Realizar una búsqueda local para mejorar la solución mediante intercambios
    de artículos.
    
    Args:
        articulos (np.ndarray): Vector de artículos disponibles
        capacidad_mochila (int): Capacidad máxima de la mochila
        mascara_seleccion (np.ndarray): Máscara booleana con los artículos
        seleccionados
        peso_actual (int): Peso actual de la mochila
        beneficio_actual (int): Beneficio actual de la mochila
        max_repeticiones (int): Número máximo de repeticiones permitidas
        umbral_mejora (int): Mínima mejora requerida para considerar un
        intercambio
        verbose (bool): Modo detallado de salida
    
    Returns:
        Dict: Resultado con la solución optimizada
    """
    ids = articulos['id']
    pesos = articulos['peso']
    beneficios = articulos['beneficio']

    # Calcular el orden de los pesos para acelerar la búsqueda
    orden_por_peso = np.argsort(pesos)
    beneficio_inicial = beneficio_actual

    # Realizar la búsqueda local
    for num_repeticion in range(1, max_repeticiones + 1):
        if verbose:
            print(f"\n--- Repetición {num_repeticion} ---")
            print(f"Beneficio actual: {beneficio_actual} "
                  f"| Peso actual: {peso_actual}/{capacidad_mochila}")
            print(f"Capacidad restante: {capacidad_mochila - peso_actual}")

        # Obtener candidatos (artículos no seleccionados) ordenados por peso
        candidatos = orden_por_peso[~mascara_seleccion[orden_por_peso]]
        if not candidatos.size:
            if verbose:
                print("No hay más candidatos para intercambio.")
            break   # Salir si no quedan candidatos

        # Preparar datos para la evaluación de intercambios
        pesos_candidatos = pesos[candidatos]
        beneficios_candidatos = beneficios[candidatos]

        # Por cada candidato, obtener el máximo beneficio existente para algún
        # candidato con menor o igual peso que el propio.
        beneficios_maximos = np.fmax.accumulate(beneficios_candidatos)

        # Encontrar posiciones donde cambia el beneficio máximo
        cambios = np.r_[
            True, beneficios_maximos[1:] != beneficios_maximos[:-1]
        ]
        posiciones_cambio = np.flatnonzero(cambios)

        # Crear un vector que le asigne a cada artículo candidato la posición
        # del candidato con peso menor o igual al propio que tenga el
        # máximo beneficio.
        mejores_indices = np.repeat(
            posiciones_cambio, np.diff(np.r_[posiciones_cambio, len(cambios)])
        )

        # Obtener artículos actualmente en la mochila
        capacidad_restante = capacidad_mochila - peso_actual
        articulos_seleccionados = np.flatnonzero(mascara_seleccion)
        pesos_seleccionados = pesos[articulos_seleccionados]

        # Para cada artículo en la mochila, calcular el peso máximo que podría
        # tener un candidato de reemplazo
        limites_peso = pesos_seleccionados + capacidad_restante

        # Para cada artículo en la mochila, encontrar la posicion máxima en el
        # vector de candidatos en la cual todos los pesos sean menores
        # al límite
        limites_superiores = np.searchsorted(
            pesos_candidatos,
            limites_peso,
            # Devuelve la posición donde se insertaría el elemento para
            # mantener el orden
            side='right'
        )

        # Crear máscara para identificar intercambios posibles (artículos
        # para los cuales hay al menos un candidato válido)
        mascara_posible_intercambio = limites_superiores > 0

        # Para cada artículo en la mochila, obtener el índice del mejor
        # candidato factible
        indices_opciones = np.where(
            mascara_posible_intercambio,
            # Obtener el último candidato válido
            mejores_indices[np.clip(limites_superiores-1, 0, None)],
            0   # Valor por defecto si no hay intercambios válidos
        )

        # alcular ganancias potenciales de cada intercambio
        ganancias_potenciales = (
            beneficios_candidatos[indices_opciones]
            - beneficios[articulos_seleccionados]
        )
        nuevos_pesos = (
            peso_actual
            - pesos_seleccionados
            + pesos_candidatos[indices_opciones]
        )

        # 3.7 Obtener sólo los intercambios factibles.
        # Un intercambio es factible si:
        # 1. La ganancia es al menos 'mejora_minima'
        # 2. El nuevo peso no excede la capacidad máxima
        mascara_intercambios_validos = (
            mascara_posible_intercambio
            & (ganancias_potenciales >= umbral_mejora)
            & (nuevos_pesos <= capacidad_mochila)
        )
        if not mascara_intercambios_validos.any():
            if verbose:
                print("No se encontraron mejoras significativas.")
            break   # Salir si no hay intercambios factibles

        # De todos los intercambios válidos, elegir el de mayor ganancia
        indice_mejor = np.argmax(
            ganancias_potenciales[mascara_intercambios_validos]
        )

        # Obtener los identificadores de los artículos involucrados en
        # el intercambio
        id_eliminar = articulos_seleccionados[
            mascara_intercambios_validos
        ][indice_mejor]
        id_agregar = candidatos[
            indices_opciones[
                mascara_intercambios_validos
            ][indice_mejor]
        ]

        # Actualizar la solución
        mascara_seleccion[id_eliminar] = False
        mascara_seleccion[id_agregar] = True
        peso_actual = nuevos_pesos[
            mascara_intercambios_validos
        ][indice_mejor]
        beneficio_actual += ganancias_potenciales[
            mascara_intercambios_validos
        ][indice_mejor]

        if verbose:
            print(f"Intercambio: Remover {id_eliminar} "
                  f"(Beneficio: {beneficios[id_eliminar]})")
            print(f"            Agregar {id_agregar} "
                  f"(Beneficio: {beneficios[id_agregar]})")
            print(f"Nuevo beneficio: {beneficio_actual} | "
                  f"Nuevo peso: {peso_actual}")

    # ----------------------------
    # 5. Preparación de resultados
    # ----------------------------
    return {
        'articulos_seleccionados':
            np.sort(ids[mascara_seleccion]).tolist(),
        'peso_total': peso_actual,
        'beneficio_total': beneficio_actual,
        'repeticiones_realizadas': num_repeticion,
        'mejora_obtenida': beneficio_actual - beneficio_inicial
    }


def resolver_problema_voraz(
        articulos: np.ndarray,
        capacidad_mochila: int,
        max_repeticiones: int = 100,
        umbral_mejora: int = 1,
        verbose: bool = False
) -> Dict[str, float | List[int] | int]:
    """
    Resolver el problema de la mochila usando un enfoque híbrido
    voraz + búsqueda local.

    Args:
        articulos (np.ndarray): Vector de artículos disponibles
        capacidad_mochila (int): Capacidad máxima de la mochila
        max_repeticiones (int): Repeticiones máximas para la búsqueda local
        umbral_mejora (int): Mínima mejora requerida para considerar un
        intercambio
        verbose (bool): Modo detallado de salida
    
    Returns:
        Dict: Mejor solución encontrada
    """
    mascara_inicial, peso_inicial, beneficio_inicial = generar_solucion_voraz(
        articulos,
        capacidad_mochila
    )
    return optimizar_solucion(
        articulos,
        capacidad_mochila,
        mascara_inicial,
        peso_inicial,
        beneficio_inicial,
        max_repeticiones,
        umbral_mejora,
        verbose
    )

def resolver_problema_grasp(
        articulos: np.ndarray,
        capacidad_mochila: int,
        repeticiones_grasp: int = 10,
        parametro_aleatoriedad: float = 0.3,
        max_repeticiones_bl: int = 100,
        umbral_mejora: int = 1,
        verbose: bool = False
) -> Dict[str, float | List[int] | int]:
    """
    Resolver el problema de la mochila usando GRASP (búsqueda aleatorizada y
    local).

    Args:
        articulos (np.ndarray): Vector de artículos disponibles
        capacidad_mochila (int): Capacidad máxima de la mochila
        repeticiones_grasp (int): Número de repeticiones de GRASP
        parametro_aleatoriedad (int): Controla la aleatoriedad
        (0=voraz, 1=aleatorio)
        umbral_mejora (int): Mínima mejora requerida para considerar un
        intercambio
        verbose (bool): Modo detallado de salida
    
    Returns:
        Dict: Mejor solución encontrada
    """
    pesos = articulos['peso']
    beneficios = articulos['beneficio']
    relacion_beneficio_peso = beneficios / pesos
    orden_descendente = np.argsort(-relacion_beneficio_peso)

    mejor_solucion = None
    mascara_seleccion = np.zeros(len(articulos), dtype=bool)
    mascara_disponibilidad = np.ones(len(articulos), dtype=bool)

    for num_repeticion in range(repeticiones_grasp):
        if verbose:
            print(f"\n=== Iteración GRASP {num_repeticion + 1}/{repeticiones_grasp} ===")

        # Fase de construcción aleatorizada
        peso_actual, beneficio_actual = 0, 0
        capacidad_restante = capacidad_mochila
        mascara_seleccion[:] = False
        mascara_disponibilidad[:] = True

        while capacidad_restante > 0:
            candidatos = orden_descendente[
                mascara_disponibilidad[orden_descendente]
            ]
            if not candidatos.size:
                break

            # Generar lista restringida de candidatos (RCL)
            relacion_candidatos = relacion_beneficio_peso[candidatos]
            max_relacion = relacion_candidatos[-1]
            min_relacion = relacion_candidatos[0]
            umbral = (max_relacion
                      - parametro_aleatoriedad
                      * (max_relacion - min_relacion))

            # Encontrar candidatos elegibles
            corte = np.searchsorted(relacion_candidatos, umbral, side='left')
            rcl = candidatos[:corte]

            if not rcl.size:
                break

            # Selección aleatoria en la RCL
            seleccion = np.random.choice(rcl)

            # Actualizar solución si es viable
            peso_item = pesos[seleccion]
            if peso_item <= capacidad_restante:
                mascara_seleccion[seleccion] = True
                peso_actual += peso_item
                beneficio_actual += beneficios[seleccion]
                capacidad_restante -= peso_item

            # Remover el artículo seleccionado de los candidatos
            mascara_disponibilidad[seleccion] = False

        # Fase de búsqueda local (usando implementación existente)
        solution = optimizar_solucion(
            articulos,
            capacidad_mochila,
            mascara_seleccion,
            peso_actual,
            beneficio_actual,
            max_repeticiones_bl,
            umbral_mejora,
            verbose
        )

        # Actualizar la mejor solución
        if not mejor_solucion or solution['beneficio_total'] > mejor_solucion['beneficio_total']:
            mejor_solucion = solution
            if verbose:
                print(f"Nueva mejor solución encontrada: {mejor_solucion['beneficio_total']}")

    return mejor_solucion

def resolver_problema_tabu(
        articulos: np.ndarray,
        capacidad_mochila: int,
        max_repeticiones: int = 100,
        tabu_tenure: int = 10,
        max_sin_mejora: int = 20,
        verbose: bool = False
) -> Dict[str, float | List[int] | int]:

    # Precomputar datos esenciales
    pesos = articulos['peso'].astype(np.int32)
    beneficios = articulos['beneficio'].astype(np.int32)
    ids = articulos['id']

    # Generar solución inicial usando el enfoque voraz
    sol_actual, peso_actual, beneficio_actual = generar_solucion_voraz(
        articulos,
        capacidad_mochila
    )
    mejor_sol = sol_actual.copy()
    mejor_peso = peso_actual
    mejor_beneficio = beneficio_actual

    # Estructuras
    lista_tabu = {}
    contador_sin_mejora = 0

    # Búsqueda principal vectorizada
    for iteracion in range(1, max_repeticiones + 1):
        if contador_sin_mejora >= max_sin_mejora:
            break

        best_mejora = -np.inf
        best_move = None
        best_peso = 0
        best_beneficio = 0

        # Movimientos de adición (vectorizado + incremental)
        no_seleccionados = np.where(~sol_actual)[0]
        if no_seleccionados.size:
            # Cálculo vectorizado de pesos y máscara factible
            nuevos_pesos = peso_actual + pesos[no_seleccionados]
            mascara_factibles = nuevos_pesos <= capacidad_mochila
            feasible_j = no_seleccionados[mascara_factibles]

            # Ordenar por beneficio descendente y buscar el mejor movimiento permitido
            if feasible_j.size:
                # Usar argsort para ordenar sin conversión a lista
                orden = np.argsort(-beneficios[feasible_j])
                sorted_j = feasible_j[orden]

                for j in sorted_j:
                    move = ('add', j)
                    mejora = beneficios[j]
                    nuevo_beneficio = beneficio_actual + mejora

                    # Evaluar si el movimiento está permitido
                    if (move not in lista_tabu or nuevo_beneficio > mejor_beneficio):
                        if mejora > best_mejora:
                            best_mejora = mejora
                            best_move = move
                            best_peso = nuevos_pesos[mascara_factibles][feasible_j == j][0]
                            best_beneficio = nuevo_beneficio
                            break  # Primer movimiento válido óptimo

        # Optimización para movimientos de intercambio
        seleccionados = np.where(sol_actual)[0]
        if seleccionados.size and no_seleccionados.size:
            # Uso de broadcasting eficiente con tipos de datos reducidos
            delta_pesos = pesos[no_seleccionados] - pesos[seleccionados[:, None]]
            factibles = (peso_actual + delta_pesos) <= capacidad_mochila

            # Encontrar índices de intercambios factibles
            i_idx, j_idx = np.where(factibles)
            if i_idx.size > 0:
                i = seleccionados[i_idx]
                j = no_seleccionados[j_idx]
                delta_ben = beneficios[j] - beneficios[i]

                # Procesar en orden descendente usando argpartition para eficiencia
                if delta_ben.size > 0:
                    max_idx = np.argmax(delta_ben)
                    current_i = i[max_idx]
                    current_j = j[max_idx]
                    move = ('swap', current_i, current_j)
                    mejora = delta_ben[max_idx]
                    nuevo_beneficio = beneficio_actual + mejora

                    if (move not in lista_tabu or nuevo_beneficio > mejor_beneficio):
                        if mejora > best_mejora:
                            best_mejora = mejora
                            best_move = move
                            best_peso = peso_actual + (pesos[current_j] - pesos[current_i])
                            best_beneficio = nuevo_beneficio

        # Aplicar mejor movimiento encontrado
        if best_move:
            # Actualizar solución
            if best_move[0] == 'add':
                sol_actual[best_move[1]] = True
            else:
                sol_actual[best_move[1]] = False
                sol_actual[best_move[2]] = True

            # Actualización de métricas
            peso_actual = best_peso
            beneficio_actual = best_beneficio

            if beneficio_actual > mejor_beneficio:
                mejor_sol[:] = sol_actual  # Actualización in-place
                mejor_peso = peso_actual
                mejor_beneficio = beneficio_actual
                contador_sin_mejora = 0
            else:
                contador_sin_mejora += 1

            # Actualizar lista tabú
            lista_tabu[best_move] = iteracion + tabu_tenure
            if best_move[0] == 'swap':
                lista_tabu[('swap', best_move[2], best_move[1])] = iteracion + tabu_tenure

        # Limpieza eficiente de lista tabú
        lista_tabu = defaultdict(int, {k: v for k, v in lista_tabu.items() if v > iteracion})

        if verbose:
            print(f"Iteración {iteracion}: Beneficio {mejor_beneficio}")

    # Cálculo final optimizado
    articulos_seleccionados = np.sort(ids[mejor_sol]).tolist()
    mejora_total = mejor_beneficio - (sol_actual @ beneficios)

    return {
        'articulos_seleccionados': articulos_seleccionados,
        'peso_total': mejor_peso,
        'beneficio_total': mejor_beneficio,
        'iteraciones_realizadas': iteracion,
        'mejora_obtenida': mejora_total
    }
