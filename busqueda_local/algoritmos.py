from typing import Dict, List, Tuple, Union
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
        duracion_tabu: int = 10,
        max_rep_sin_mejora: int = 20,
        mostrar_progreso: bool = False
) -> Dict[str, Union[float, List[int], int]]:
    """
    Resolver el problema de la mochila usando búsqueda tabú.

    Args:
        articulos: Vector estructurado con campos 'id', 'peso' y 'beneficio'.
        capacidad_mochila: Capacidad máxima de carga de la mochila
        max_repeticiones: Máximo número de repeticiones permitidas
        duracion_tabu: Número de repeticiones que un movimiento permanece en la
        lista tabú.
        max_rep_sin_mejora: Criterio de parada por estancamiento.
        mostrar_progreso: Si es verdadero, muestra el progreso durante la
        ejecución.

    Returns:
        Diccionario con la solución óptima encontrada y métricas relevantes.
    """

    # 1. Preparación inicial de datos

    # Extraer y convertir datos a tipos eficientes para operaciones numéricas
    pesos = articulos['peso'].astype(np.int32)
    beneficios = articulos['beneficio'].astype(np.int32)
    ids_articulos = articulos['id']

    # Generar solución inicial usando algoritmo voraz
    solucion_actual, peso_actual, beneficio_actual = generar_solucion_voraz(
        articulos, capacidad_mochila)

    # Asignar la solución inicial como mejor solución
    mejor_solucion = solucion_actual.copy()
    mejor_peso = peso_actual
    mejor_beneficio = beneficio_actual

    # 2. Configuración de estructuras para la búsqueda tabú

    lista_tabu = {} # Diccionario para registrar movimientos prohibidos
    contador_sin_mejora = 0 # Repeticiones consecutivas sin mejoras

    # 3. Ciclo principal de la búsqueda tabú

    for repeticion_actual in range(1, max_repeticiones + 1):
        # Criterio de interrupción temprana por estancamiento
        if contador_sin_mejora >= max_rep_sin_mejora:
            break

        # Variables para rastrear el mejor movimiento en la repetición actual
        maxima_mejora = -np.inf
        mejor_movimiento = None
        mejor_peso_candidato = 0
        mejor_beneficio_candidato = 0

        # 3.1 Evaluación de movimientos de adición

        # Identificar artículos no seleccionados que podrían añadirse
        indices_no_seleccionados = np.where(~solucion_actual)[0]

        if indices_no_seleccionados.size:
            # Calcular nuevos pesos potenciales de forma vectorizada
            pesos_candidatos = peso_actual + pesos[indices_no_seleccionados]

            # Crear máscar para movimientos factibles (que no excedan la
            #  capacidad)
            mascara_factibles = pesos_candidatos <= capacidad_mochila
            indices_factibles = indices_no_seleccionados[mascara_factibles]

            if indices_factibles.size:
                # Ordenar candidatos por beneficio descendente para evaluación
                # ordenada
                orden_descendente = np.argsort(-beneficios[indices_factibles])
                candidatos_ordenados = indices_factibles[orden_descendente]

                # Evaluar cada movimiento de adición por orden de beneficio
                for indice_articulo in candidatos_ordenados:
                    tipo_movimiento = ('añadir', indice_articulo)
                    mejora_potencial = beneficios[indice_articulo]
                    beneficio_candidato = beneficio_actual + mejora_potencial

                    # Verificar si el movimiento está permitido (reglas tabú y
                    # de aspiración)
                    movimiento_permitido = (
                        tipo_movimiento not in lista_tabu or
                        beneficio_candidato > mejor_beneficio
                    )
                    if (movimiento_permitido and
                        mejora_potencial > maxima_mejora):
                        maxima_mejora = mejora_potencial
                        mejor_movimiento = tipo_movimiento
                        mejor_peso_candidato = pesos_candidatos[
                            mascara_factibles
                        ][indices_factibles == indice_articulo][0]
                        mejor_beneficio_candidato = beneficio_candidato
                        break  # Usar el primer mejor movimiento encontrado

        # 3.2 Evaluación de movimientos de intercambio

        indices_seleccionados = np.where(solucion_actual)[0]
        if indices_seleccionados.size and indices_no_seleccionados.size:
            # Calcular diferencias de peso para todos los posibles intercambios
            # usando difusión
            delta_pesos = (pesos[indices_no_seleccionados] -
                pesos[indices_seleccionados[:, None]])
            intercambios_factibles = ((peso_actual + delta_pesos) <=
                capacidad_mochila)

            # Obtener índices de todos los intercambios posibles que sean
            # factibles
            filas, columnas = np.where(intercambios_factibles)

            if filas.size > 0:
                # Calcular beneficios netos para cada intercambio factible
                articulos_salida = indices_seleccionados[filas]
                articulos_entrada = indices_no_seleccionados[columnas]
                delta_beneficios = (beneficios[articulos_entrada] -
                    beneficios[articulos_salida])

                # Encontrar el intercambio con máxima mejora
                indice_maximo = np.argmax(delta_beneficios)
                articulo_remover = articulos_salida[indice_maximo]
                articulo_agregar = articulos_entrada[indice_maximo]
                tipo_movimiento = ('intercambiar',
                    articulo_remover, articulo_agregar)
                mejora_potencial = delta_beneficios[indice_maximo]
                beneficio_candidato = beneficio_actual + mejora_potencial

                # Verificar restricciones tabú
                movimiento_permitido = (
                    tipo_movimiento not in lista_tabu or
                    beneficio_candidato > mejor_beneficio
                )

                if movimiento_permitido and mejora_potencial > maxima_mejora:
                    maxima_mejora = mejora_potencial
                    mejor_movimiento = tipo_movimiento
                    mejor_peso_candidato = peso_actual + (
                        pesos[articulo_agregar] - pesos[articulo_remover])
                    mejor_beneficio_candidato = beneficio_candidato

        # 4. Aplicar el mejor movimiento encontrado

        if mejor_movimiento:
            # Actualizar la solución actual según el tipo de movimiento
            if mejor_movimiento[0] == 'añadir':
                solucion_actual[mejor_movimiento[1]] = True
            else:
                # Para intercambios quitar un artículo y añadir otro
                solucion_actual[mejor_movimiento[1]] = False
                solucion_actual[mejor_movimiento[2]] = True

            # Actualizar métricas de la solución actual
            peso_actual = mejor_peso_candidato
            beneficio_actual = mejor_beneficio_candidato

            # Actualizar mejor solución global si hay mejora
            if beneficio_actual > mejor_beneficio:
                mejor_solucion[:] = solucion_actual
                mejor_peso = peso_actual
                mejor_beneficio = beneficio_actual
                contador_sin_mejora = 0  # Reiniciar contador de estancamiento
            else:
                contador_sin_mejora += 1

            # Registrar movimiento en lista tabú
            lista_tabu[mejor_movimiento] = repeticion_actual + duracion_tabu

            # Para intercambios, registrar también el movimiento inverso
            if mejor_movimiento[0] == 'intercambiar':
                movimiento_inverso = ('intercambiar',
                    mejor_movimiento[2], mejor_movimiento[1])
                lista_tabu[movimiento_inverso] = (repeticion_actual +
                    duracion_tabu)

        # 5. Mantenimiento de la lista tabú

        # Eliminar entradas caducadas de la lista tabú
        lista_tabu = {movimiento: expiracion
            for movimiento, expiracion in lista_tabu.items()
            if expiracion > repeticion_actual}

        # Mostrar progreso (opcional)
        if mostrar_progreso:
            print(f"Iteración {repeticion_actual}: " +
                f"Beneficio actual {mejor_beneficio}")

    # 6. Preparación de resultados finales

    # Obtener los identificadores ordenados de los artículos seleccionados
    articulos_seleccionados = np.sort(ids_articulos[mejor_solucion]).tolist()

    # Calcular mejora total respecto a la solución inicial
    mejora_total = mejor_beneficio - (solucion_actual @ beneficios)

    return {
        'articulos_seleccionados': articulos_seleccionados,
        'peso_total': mejor_peso,
        'beneficio_total': mejor_beneficio,
        'iteraciones_realizadas': repeticion_actual,
        'mejora_obtenida': mejora_total
    }
