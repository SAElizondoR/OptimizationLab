import numpy as np

def cargar_datos(ruta_archivo):
    """Cargar los datos del problema desde un archivo CSV.
    
    Args:
        ruta_archivo (str): Ruta del archivo CSV con los datos
    
    Returns:
        np.ndarray: Matriz con los datos de los artículos.
    """
    tipo_datos = np.dtype([
        ('id', 'u2'),   # Identificador único del artículo
        ('peso', 'i4'),  # Peso del artículo (número entero de 4 octetos)
        ('beneficio', 'i4') # Beneficio del artículo
    ])

    return np.loadtxt(
        ruta_archivo,
        delimiter=',',
        skiprows=1,
        dtype=tipo_datos
    )

def resolver_mochila(datos, capacidad_maxima, max_repeticiones=100,
                     mejora_minima=1, verbose=False):
    """
    Resolver el problema de la mochila usando un enfoque híbrido
    voraz + búsqueda local.

    Args:
        datos (np.ndarray): Datos de los artículos
        capacidad_maxima (int): Capacidad máxima de la mochila
        max_repeticiones (int): Máximo número de repeticiones para la
            búsqueda local
        mejora_minima (int): Umbral mínimo de mejora para continuar la búsqueda
        verbose (bool): Mostrar el progreso en la terminal
    """
    # ----------------------------
    # 1. Configuración inicial
    # ----------------------------
    ids_articulos = datos['id']
    pesos_articulos = datos['peso']
    beneficios_articulos = datos['beneficio']
    total_articulos = len(datos)

    # ----------------------------
    # 2. Solución inicial (enfoque voraz)
    # ----------------------------
    # Calcular la relación beneficio/peso
    tasa_beneficio_peso = -beneficios_articulos / pesos_articulos

    # Ordenar los artículos descendentemente de acuerdo con su
    # relación beneficio/peso
    indices_ordenados_por_tasa = np.lexsort(
        (pesos_articulos, tasa_beneficio_peso)
    )

    # Seleccionar artículos hasta llenar la capacidad de la mochila
    pesos_acumulados = np.cumsum(pesos_articulos[indices_ordenados_por_tasa])
    total_articulos_validos = np.searchsorted(
        pesos_acumulados,
        capacidad_maxima,
        side='right'
    )

    # Crear máscara con selección inicial
    mascara_seleccionados = np.zeros(total_articulos, dtype=bool)
    mascara_seleccionados[
        indices_ordenados_por_tasa[:total_articulos_validos]
    ] = True

    # Calcular métricas iniciales
    peso_actual = (pesos_acumulados[total_articulos_validos-1]
                   if total_articulos_validos > 0 else 0)
    beneficio_inicial = beneficios_articulos[
        indices_ordenados_por_tasa[:total_articulos_validos]
    ].sum()
    beneficio_actual = beneficio_inicial

    if verbose:
        print("\n--- Fase Voraz Inicial ---")
        print("Artículos seleccionados: "
              f"{ids_articulos[mascara_seleccionados].tolist()}")
        print(f"Peso total: {peso_actual}/{capacidad_maxima}")
        print(f"Beneficio inicial: {beneficio_inicial}")
    
    # ----------------------------
    # 3. Búsqueda local
    # ----------------------------
    # Ordenar artículos ascendentemente por peso
    indices_ordenados_por_peso = np.argsort(pesos_articulos)

    # Realizar la búsqueda local
    for numero_repeticion in range(1, max_repeticiones + 1):
        if verbose:
            print(f"\n--- Repetición {numero_repeticion} ---")
            print(f"Beneficio actual: {beneficio_actual} "
                  f"| Peso actual: {peso_actual}")
            print(f"Capacidad restante: {capacidad_maxima - peso_actual}")
        
        # 3.1 Identificar candidatos potenciales (artículos no seleccionados)
        # ---------------------------------------------------------------------
        candidatos = indices_ordenados_por_peso[
            ~mascara_seleccionados[indices_ordenados_por_peso]
        ]
        if not candidatos.size:
            if verbose: print("No hay más candidatos para intercambio.")
            break   # Salir si no quedan candidatos

        pesos_candidatos = pesos_articulos[candidatos]
        beneficios_candidatos = beneficios_articulos[candidatos]

        # 3.2 Encontrar los mejores candidatos por rango de peso
        # ---------------------------------------------------------------------
        # Por cada candidato, obtener el máximo beneficio existente para algún
        # candidato con menor o igual peso que el propio.
        beneficios_maximos = np.fmax.accumulate(beneficios_candidatos)

        # Encontrar posiciones donde cambia el beneficio máximo
        posiciones_cambio_maximo = np.flatnonzero(
            np.r_[True, beneficios_maximos[1:] != beneficios_maximos[:-1]]
        )

        # Crear un vector que le asigne a cada artículo candidato la posición
        # del candidato con peso menor o igual al propio que tenga el
        # máximo beneficio.
        indices_mejores_candidatos = np.repeat(
            posiciones_cambio_maximo,
            np.diff(np.r_[posiciones_cambio_maximo, len(beneficios_maximos)])
        )

        # 3.3 Preparar los datos para hacer intercambios
        # ---------------------------------------------------------------------
        capacidad_restante = capacidad_maxima - peso_actual

        # Artículos actualmente en la mochila
        ids_seleccionados = np.flatnonzero(mascara_seleccionados)
        pesos_seleccionados = pesos_articulos[ids_seleccionados]

        # 3.4 Búsqueda de intercambios válidos
        # ---------------------------------------------------------------------
        # Para cada artículo en la mochila, calcular el peso máximo que podría
        # tener un candidato de reemplazo
        pesos_limite = pesos_seleccionados + capacidad_restante

        # Para cada artículo en la mochila, encontrar la posicion máxima en el
        # vector de candidatos en la cual todos los pesos sean menores
        # al límite
        limites_superiores = np.searchsorted(
            pesos_candidatos,
            pesos_limite,
            # Devuelve la posición donde se insertaría el elemento para
            # mantener el orden
            side='right'
        )

        # Crear máscara para identificar intercambios posibles (artículos
        # para los cuales hay al menos un candidato válido)
        mascara_intercambios_validos = limites_superiores > 0

        # 3.5 Determinar la mejor opción de intercambio para cada artículo
        # seleccionado
        # ---------------------------------------------------------------------
        # Para cada artículo en la mochila, obtener el índice del mejor
        # candidato factible
        indices_opciones = np.where(
            mascara_intercambios_validos,
            # Obtener el último candidato válido
            indices_mejores_candidatos[np.clip(limites_superiores-1, 0, None)],
            0   # Valor por defecto si no hay intercambios válidos
        )

        # 3.6 Calcular ganancias potenciales de cada intercambio
        # ---------------------------------------------------------------------
        ganancias_potenciales = (
            beneficios_candidatos[indices_opciones]
            - beneficios_articulos[ids_seleccionados]
        )
        nuevos_pesos = (
            peso_actual
            - pesos_seleccionados
            + pesos_candidatos[indices_opciones]
        )

        # 3.7 Obtener sólo los intercambios factibles
        # ---------------------------------------------------------------------
        # Un intercambio es factible si:
        # 1. La ganancia es al menos 'mejora_minima'
        # 2. El nuevo peso no excede la capacidad máxima
        mascara_intercambios_validos &= (
            (ganancias_potenciales >= mejora_minima)
            & (nuevos_pesos <= capacidad_maxima)
        )
        if not mascara_intercambios_validos.any():
            if verbose: print("No se encontraron mejoras significativas.")
            break   # Salir si no hay intercambios factibles

        # 3.8 Aplicar el mejor intercambio
        # ---------------------------------------------------------------------
        # De todos los intercambios válidos, elegir el de mayor ganancia
        indice_mejor_ganancia = np.argmax(
            ganancias_potenciales[mascara_intercambios_validos]
        )

        # Obtener los identificadores de los artículos involucrados en
        # el intercambio
        id_articulo_remover = ids_seleccionados[
            mascara_intercambios_validos
        ][indice_mejor_ganancia]
        id_articulo_agregar = candidatos[
            indices_opciones[
                mascara_intercambios_validos
            ][indice_mejor_ganancia]
        ]

        # Actualizar la solución
        mascara_seleccionados[id_articulo_remover] = False
        mascara_seleccionados[id_articulo_agregar] = True
        peso_actual = nuevos_pesos[
            mascara_intercambios_validos
        ][indice_mejor_ganancia]
        beneficio_actual += ganancias_potenciales[
            mascara_intercambios_validos
        ][indice_mejor_ganancia]

        if verbose:
            print(f"Intercambio: Remover {id_articulo_remover} "
                  f"(Beneficio: {beneficios_articulos[id_articulo_remover]})")
            print(f"            Agregar {id_articulo_agregar} "
                  f"(Beneficio: {beneficios_articulos[id_articulo_agregar]})")
            print(f"Nuevo beneficio: {beneficio_actual} | "
                  f"Nuevo peso: {peso_actual}")
    
    # ----------------------------
    # 5. Preparación de resultados
    # ----------------------------
    return {
        'articulos_seleccionados':
            np.sort(ids_articulos[mascara_seleccionados]).tolist(),
        'peso_total': peso_actual,
        'beneficio_total': beneficio_actual,
        'repeticiones_realizadas': numero_repeticion,
        'mejora_obtenida': beneficio_actual - beneficio_inicial
    }

# ----------------------------
# Ejemplo de uso
# ----------------------------
if __name__ == "__main__":
    # Configuración de prueba
    ARCHIVO_DATOS = "datos_dificil.csv"
    CAPACIDAD_MOCHILA = 1000000

    # Cargar datos y resolver
    dataset = cargar_datos(ARCHIVO_DATOS)
    solucion = resolver_mochila(
        datos=dataset,
        capacidad_maxima=CAPACIDAD_MOCHILA,
        max_repeticiones=100,
        verbose=True
    )

    # Mostrar resultados
    print("\n--- Resultado Final ---")
    print("Artículos seleccionados: "
          f"{len(solucion['articulos_seleccionados'])}")
    print(f"Peso total: {solucion['peso_total']}/{CAPACIDAD_MOCHILA}")
    print(f"Beneficio total: {solucion['beneficio_total']}")
    print(f"Mejora obtenida: {solucion['mejora_obtenida']}")
    print("Número de repeticiones realizadas: "
          f"{solucion['repeticiones_realizadas']}")
