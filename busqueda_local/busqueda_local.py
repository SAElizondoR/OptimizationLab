import time
import argparse
import numpy as np
from algoritmos import (resolver_problema_voraz, resolver_problema_grasp,
                        resolver_problema_tabu)

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
    parser.add_argument('-t', '--tabu', action='store_true',
                        help='Usar algoritmo de búsqueda tabú')
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

        # Seleccionar el algoritmo a utilizar
        if args.tabu:
            print("\nUsando algoritmo de búsqueda tabú")
            solucion = resolver_problema_tabu(
                articulos=dataset,
                capacidad_mochila=args.capacidad,
                max_repeticiones=args.repeticiones,
                mostrar_progreso=args.verbose
            )
        elif args.grasp:
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
                articulos=dataset,
                capacidad_mochila=args.capacidad,
                max_repeticiones=args.repeticiones,
                verbose=args.verbose
            )

        tiempo_ejecucion = time.perf_counter() - inicio_ejecucion

        # Tiempo total
        tiempo_total = tiempo_carga + tiempo_ejecucion

        # Mostrar resultados
        print("\n" + "="*50)
        algoritmo = "Búsqueda Tabú" if args.tabu else ("GRASP" if args.grasp else " Voraz + Búsqueda Local")
        print(f"Resultados (Algoritmo: {algoritmo})".center(50))
        print("="*50)
        print(f"Archivo procesado: {args.archivo}")
        print(f"Capacidad configurada: {args.capacidad}")
        print(f"\nArtículos seleccionados: {len(solucion['articulos_seleccionados'])}")
        print(f"Peso total: {solucion['peso_total']}/{args.capacidad}")
        print(f"Beneficio total: {solucion['beneficio_total']}")
        print(f"Mejora obtenida: {solucion['mejora_obtenida']}")
        # print(f"Repeticiones realizadas: {solucion['repeticiones_realizadas']}")

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
