import numpy as np

def generar_instancia(filename):
    """
    Generar un archivo CSV con una instancia compleja del problema de la
    mochila. El archivo contiene 10,000 artículos con diferentes patrones de
    pesos y beneficios.
    """
    buffer_size = 10000 # Tamaño para 10,000 artículos
    buffer = np.empty(buffer_size + 1, dtype='U25')
    ptr = 0 # Apuntador para rastrear la posición actual en la memoria

    # 1. Cabecera del archivo
    buffer[ptr] = 'id,peso,beneficio'
    ptr += 1

    # 2. Artículos especiales (casos límite)
    specials = np.array([
        "1,999999,999999",  # Artículo trampa: Peso y beneficio máximos
        "16,1000,2000", "17,2000,1999", "18,1500,3000", # Pares trampa
        "19,3000,2999", "20,750,1500", "21,1500,1499",
        "9998,1,100", "9999,2,199", "10000,3,297"
    ], dtype='U25')

    # Copiar los artículos a la memoria
    buffer[ptr:ptr+len(specials)] = specials
    ptr += len(specials)

    # 3. Patrón geométrico (pesos y beneficios decrecientes)
    i_vals = np.arange(2, 16)   # Artículos del 2 al 15
    weights = 500000 >> (i_vals - 2)    # Pesos geométricamente decrecientes
    beneficios = (weights * 1.2).astype(int)

    # Construir las líneas del archivo
    geometric = np.core.defchararray.add(
        np.core.defchararray.add(
            np.core.defchararray.add(i_vals.astype('U4'), ','), # Id.
            weights.astype('U6')    # Peso
        ),
        np.core.defchararray.add(',', beneficios.astype('U6'))  # Beneficio
    )
    buffer[ptr:ptr+len(geometric)] = geometric
    ptr += len(geometric)
    
    # 4. Zona crítica (artículos con máxima relación beneficio-peso)
    crit_ids = np.arange(5000, 5003)    # Artículos del 5000 al 5002
    crit_weights = 100 - (crit_ids - 5000)  # Pesos decrecientes
    crit_benef = 10000 - (crit_ids - 5000)  # Beneficios decrecientes

    critical = np.core.defchararray.add(
        np.core.defchararray.add(
            np.core.defchararray.add(crit_ids.astype('U4'), ','),
            crit_weights.astype('U3')
        ),
        np.core.defchararray.add(',', crit_benef.astype('U5'))
    )
    buffer[ptr:ptr+len(critical)] = critical
    ptr += len(critical)

    # 5. Generación de artículos aleatorios
    mid_ids = np.concatenate((
        np.arange(22, 5000),    # Artículos del 22 al 4999
        np.arange(5003, 9998)   # Artículos del 5003 al 9997
    ))
    weights = np.random.randint(10, 500, mid_ids.size)
    relacion = np.clip(1.5 - (mid_ids/10000), 0.8, 1.5)
    beneficios = (weights * relacion).astype(int)   # Beneficios calculados
    
    randoms = np.core.defchararray.add(
        np.core.defchararray.add(
            np.core.defchararray.add(mid_ids.astype('U5'), ','),
            weights.astype('U3')
        ),
        np.core.defchararray.add(',', beneficios.astype('U4'))
    )
    buffer[ptr:ptr+len(randoms)] = randoms
    ptr += len(randoms)

    # 6. Escritura en disco
    with open(filename, 'w') as f:
        f.write('\n'.join(buffer[:ptr]))

# Generar el archivo CSV
generar_instancia("datos_dificil.csv")
