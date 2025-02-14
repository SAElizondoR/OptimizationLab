import numpy as np
import argparse
import time
from typing import Dict, List, Tuple

def cargar_datos(ruta_archivo):
    """Cargar los datos del problema desde un archivo CSV.
    
    Args:
        ruta_archivo (str): Ruta al archivo CSV con los datos del problema
    
    Returns:
        np.ndarray: Vector estructurado con los campos id, peso y beneficio.
    """
    # Definición del tipo de datos estructurado para los artículos
    tipo_datos = np.dtype([
        ('id', 'u2'),   # Identificador único (entero sin signo de 2 octetos)
        ('peso', 'i4'),  # Peso del artículo (entero de 4 octetos)
        ('beneficio', 'i4') # Beneficio del artículo (entero de 4 octetos)
    ])

    return np.loadtxt(
        ruta_archivo,
        delimiter=',',
        skiprows=1, # Salta encabezado
        dtype=tipo_datos,
        encoding='utf-8'
    )

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
            if verbose: print("No se encontraron mejoras significativas.")
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

def main():
    # Configurar el analizador sintáctico
    parser = argparse.ArgumentParser(description='Problema de la mochila')
    parser.add_argument('-f', '--archivo', type=str, required=True,
                        help='Ruta al archivo CSV con los datos')
    parser.add_argument('-c', '--capacidad', type=int, required=True,
                        help='Capacidad máxima de la mochila')
    parser.add_argument('-r', '--repeticiones', type=int, default=100,
                        help='Número máximo de repeticiones (default: 100)')
    parser.add_argument('-a', '--alpha', type=float, default=0.3,
                        help='Parámetro alpha para GRASP (default: 0.3)')
    parser.add_argument('-g', '--grasp', action='store_true',
                        help='Usar algoritmo GRASP en lugar de voraz + búsqueda local')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Mostrar detalles durante la ejecución')
    
    args = parser.parse_args()

    # Validación de la entrada
    if args.capacidad <= 0:
        raise ValueError("La capacidad debe ser un número positivo")
    if args.grasp and (args.alpha < 0 or args.alpha > 1):
        raise ValueError("El parámetro alpha debe estar entre 0 y 1")
    
    try:
        # Medición del tiempo total
        inicio = time.perf_counter()

        # Cargar los datos
        dataset = cargar_datos(args.archivo)
        inicio_ejecucion = time.perf_counter()
        tiempo_carga = inicio_ejecucion - inicio

        # Resolver el problema con el enfoque seleccionado
        if args.grasp:
            print("\nUsando el algoritmo GRASP")
            solucion = resolver_problema_grasp(
                articulos=dataset,
                capacidad_mochila=args.capacidad,
                repeticiones_grasp=args.repeticiones,
                parametro_aleatoriedad=args.alpha,
                max_repeticiones_bl=args.repeticiones,
                verbose=args.verbose
            )
        else:
            print("\nUsando enfoque voraz + búsqueda local")
            solucion = resolver_problema_voraz(
                datos=dataset,
                capacidad=args.capacidad,
                max_repeticiones=args.repeticiones,
                verbose=args.verbose
            )
        
        tiempo_ejecucion = time.perf_counter() - inicio_ejecucion

        # Tiempo total
        tiempo_total = tiempo_carga + tiempo_ejecucion

        # Mostrar resultados
        print("\n" + "="*50)
        print(f"Resultados (Algoritmo: {'GRASP' if args.grasp else 'Voraz + Búsqueda Local'})".center(50))
        print("="*50)
        print(f"Archivo procesado: {args.archivo}")
        print(f"Capacidad configurada: {args.capacidad}")
        print(f"\nArtículos seleccionados: {len(solucion['articulos_seleccionados'])}")
        print(f"Peso total: {solucion['peso_total']}/{args.capacidad}")
        print(f"Beneficio total: {solucion['beneficio_total']}")
        print(f"Mejora obtenida: {solucion['mejora_obtenida']}")
        print(f"Repeticiones realizadas: {solucion['repeticiones_realizadas']}")

        print("\n" + "-"*50)
        print("Tiempos de ejecución:".center(50))
        print("-"*50)
        print(f"Carga de datos: {tiempo_carga:.4f} segundos")
        print(f"Algoritmo mochila: {tiempo_ejecucion:.4f} segundos")
        print(f"TOTAL: {tiempo_total:.4f} segundos")
        print("="*50)
    
    except FileNotFoundError:
        print(f"Error: El archivo {args.archivo} no existe")
    except Exception as e:
        print(f"Error inesperado: {str(e)}")

# ----------------------------
# Ejemplo de uso
# ----------------------------
if __name__ == "__main__":
    main()
