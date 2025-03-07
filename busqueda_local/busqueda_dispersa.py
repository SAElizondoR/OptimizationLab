from typing import Dict
import random
import numpy as np
from algoritmos import generar_solucion_voraz, resolver_problema_voraz, resolver_problema_grasp

def resolver_problema_scatter(
    articulos: np.ndarray,
    capacidad_mochila: int,
    max_repeticiones: int = 100,
    tamano_referencia: int = 10,
    verbose: bool = False
) -> Dict:
    """Implementación optimizada de Scatter Search usando funciones existentes."""
    # 1. Generar población inicial usando métodos optimizados
    poblacion = _generar_poblacion_inicial(
        articulos,
        capacidad_mochila,
        tamano_poblacion=100
    )

    # 2. Construir conjunto de referencia
    referencia = _construir_conjunto_referencia(poblacion, tamano_referencia)

    mejor_beneficio = -np.inf
    estancamientos = 0

    for repeticion in range(max_repeticiones):
        # 3. Generar nuevos candidatos combinando soluciones
        nuevos_candidatos = _generar_candidatos(referencia, articulos, capacidad_mochila)

        # 4. Actualizar conjunto de referencia
        referencia = _actualizar_referencia(referencia + nuevos_candidatos, tamano_referencia)

        # 5. Verificar mejora
        mejor_actual = max(sol['beneficio_total'] for sol in referencia)
        if mejor_actual > mejor_beneficio:
            mejor_beneficio = mejor_actual
            estancamientos = 0
        else:
            estancamientos += 1

        if verbose:
            print(f"Repetición {repeticion+1}: Mejor beneficio = {mejor_beneficio}")

        if estancamientos >= 10:
            break

    return max(referencia, key=lambda x: x['beneficio_total'])

# Funciones auxiliares optimizadas
def _generar_poblacion_inicial(articulos, capacidad, tamano_poblacion=100):
    """Genera población usando métodos GRASP y voraz optimizados."""
    poblacion = []
    for _ in range(tamano_poblacion):
        if random.random() < 0.5:
            sol = resolver_problema_grasp(
                articulos, capacidad,
                repeticiones_grasp=1,
                parametro_aleatoriedad=0.5,
                max_repeticiones_bl=10,
                verbose=False
            )
        else:
            sol = resolver_problema_voraz(articulos, capacidad, max_repeticiones=0)
        poblacion.append(sol)
    return poblacion

def _construir_conjunto_referencia(poblacion, tamano_ref):
    """Selección optimizada de referencia con diversidad."""
    poblacion_ordenada = sorted(poblacion, key=lambda x: x['beneficio_total'], reverse=True)
    referencia = poblacion_ordenada[:tamano_ref//2]

    # Selección diversificada usando numpy para eficiencia
    mascaras = [set(sol['articulos_seleccionados']) for sol in referencia]
    for sol in poblacion_ordenada[tamano_ref//2:]:
        if len(referencia) >= tamano_ref:
            break
        sol_set = set(sol['articulos_seleccionados'])
        if not any(len(sol_set & ref_set)/len(sol_set) > 0.3 for ref_set in mascaras):
            referencia.append(sol)
            mascaras.append(sol_set)

    return referencia

def _generar_candidatos(referencia, articulos, capacidad):
    """Genera candidatos combinando soluciones de referencia."""
    nuevos_candidatos = []
    for i in range(len(referencia)):
        for j in range(i+1, len(referencia)):
            # Combinación optimizada usando operaciones de máscara
            mask1 = _solucion_a_mascara(referencia[i], articulos)
            mask2 = _solucion_a_mascara(referencia[j], articulos)
            combinada = mask1 | mask2
            sol_combinada = _reparar_solucion(combinada, articulos, capacidad)
            nuevos_candidatos.append(sol_combinada)
    return nuevos_candidatos

def _reparar_solucion(mascara, articulos, capacidad):
    """Reparación usando función voraz."""
    peso_total = (articulos['peso'] * mascara).sum()
    if peso_total <= capacidad:
        return _mascara_a_solucion(mascara, articulos)

    # Usar método voraz para reparación
    sol_voraz = generar_solucion_voraz(articulos[mascara], capacidad)
    return _mascara_a_solucion(sol_voraz[0], articulos)

def _actualizar_referencia(candidatos, tamano_ref):
    """Selección de referencia."""
    candidatos_ordenados = sorted(candidatos, key=lambda x: x['beneficio_total'], reverse=True)
    return candidatos_ordenados[:tamano_ref]

# Funciones de conversión
def _solucion_a_mascara(solucion, articulos):
    """Convierte solución a máscara booleana."""
    mascara = np.zeros(len(articulos), dtype=bool)
    mascara[np.searchsorted(articulos['id'], solucion['articulos_seleccionados'])] = True
    return mascara

def _mascara_a_solucion(mascara, articulos):
    """Convierte máscara a formato de solución."""
    seleccionados = articulos[mascara]['id'].tolist()
    return {
        'articulos_seleccionados': seleccionados,
        'peso_total': (articulos['peso'] * mascara).sum(),
        'beneficio_total': (articulos['beneficio'] * mascara).sum(),
        'mejora_obtenida': 0
    }
