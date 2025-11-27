#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracción de imágenes desde archivos .mat del dataset CASIA-D.

DESCRIPCIÓN:
    Este módulo procesa archivos .mat de MATLAB que contienen datos del dataset
    CASIA-D (Chinese Academy of Sciences' Institute of Automation - Database D).
    Extrae imágenes de huellas plantares (footprint) o patrones de marcha (gait)
    y las guarda como archivos PNG individuales.

DATASET CASIA-D:
    - Contiene datos de presión plantar de múltiples sujetos
    - Formato: Archivos .mat de MATLAB con estructura anidada
    - Tipos de datos: 'footprint' (huellas estáticas) y 'gait' (marcha dinámica)
    - Cada registro incluye: ID del sujeto e imagen de presiones

USO EN EL PROYECTO:
    - Opción 1 de main.py: "Extraer imágenes desde archivo .mat (CASIA)"
    - Permite procesar CASfootprint.mat o CASgait.mat
    - Guarda todas las imágenes con nombres descriptivos: sujeto_X_registro_Y.png

FUNCIONES PRINCIPALES:
    - procesar_casia_mat(): Función principal que extrae y guarda todas las imágenes

ESTRUCTURA DEL ARCHIVO .MAT:
    mat_file
    └── variable_principal (array de registros)
        └── registro[i]
            ├── id: ID del sujeto
            └── footprint/gait
                └── img/gait: Array 2D con la imagen de presiones

EJEMPLO DE SALIDA:
    Input:  CASfootprint.mat
    Output: sujeto_1_registro_0.png
            sujeto_1_registro_1.png
            sujeto_2_registro_0.png
            ...
"""

import os
from pathlib import Path
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def procesar_casia_mat(ruta_archivo: str, carpeta_salida: str, guardar_todas: bool = True):
    """
    Procesa un archivo .mat de CASIA y extrae todas las imágenes de presiones plantares.

    Esta función es el punto de entrada principal para el procesamiento de archivos
    .mat del dataset CASIA-D. Automáticamente detecta la estructura del archivo,
    identifica si contiene datos de 'footprint' o 'gait', y extrae todas las imágenes.

    Args:
        ruta_archivo (str): Ruta completa al archivo .mat (ej. "CASfootprint.mat" o "CASgait.mat")
        carpeta_salida (str): Ruta a la carpeta donde se guardarán las imágenes PNG extraídas
        guardar_todas (bool): 
            - True (por defecto): Extrae y guarda todas las imágenes encontradas
            - False: Solo muestra la primera imagen como vista previa (no guarda nada)

    Proceso:
        1. Valida que el archivo .mat existe
        2. Carga el archivo usando scipy.io.loadmat
        3. Detecta automáticamente la variable principal de datos
        4. Itera sobre todos los registros
        5. Para cada registro:
           - Extrae el ID del sujeto
           - Busca la estructura 'footprint' o 'gait'
           - Extrae la imagen de presiones
           - Guarda como PNG con nombre descriptivo
        6. Reporta el número total de imágenes guardadas

    Formato de nombres de salida:
        sujeto_{id_sujeto}_registro_{numero}.png
        Ejemplo: sujeto_9_registro_876.png

    Manejo de errores:
        - Registros con estructura inválida se omiten con un warning
        - Campos faltantes se manejan graciosamente
        - Errores críticos se reportan y detienen el proceso

    Examples:
        >>> # Extraer todas las imágenes
        >>> procesar_casia_mat("CASfootprint.mat", "output/casia_images/")
        
        >>> # Solo vista previa de la primera imagen
        >>> procesar_casia_mat("CASgait.mat", "output/", guardar_todas=False)
    """
    # --- 1. Validaciones Iniciales ---
    p_archivo = Path(ruta_archivo)
    p_salida = Path(carpeta_salida)

    if not p_archivo.exists():
        print(f"[ERROR] El archivo no se encontró en la ruta: {p_archivo}")
        return

    if guardar_todas and not p_salida.exists():
        p_salida.mkdir(parents=True, exist_ok=True)
        print(f"Carpeta de salida creada: '{p_salida}'")

    # --- 2. Carga y Análisis del Archivo .mat ---
    try:
        mat_contents = sio.loadmat(str(p_archivo), squeeze_me=True)
        print(f"--- Analizando archivo: {p_archivo.name} ---")

        # Búsqueda automática de la variable principal de datos
        variables = [k for k in mat_contents.keys() if not k.startswith('__')]
        if not variables:
            print("[ERROR] No se encontraron variables de datos en el archivo.")
            return

        nombre_variable = variables[0]
        datos_principales = mat_contents[nombre_variable]
        print(f"Variable de datos encontrada: '{nombre_variable}' (contiene {datos_principales.size} registros)")

        # --- 3. Extracción y Procesamiento de Imágenes ---
        contador_guardadas = 0
        for i, registro in enumerate(np.ravel(datos_principales)):
            try:
                # Extraer el ID del sujeto de forma segura
                id_sujeto = registro['id']

                # Detectar el tipo de dato (huella o marcha) y extraerlo
                campo_intermedio = None
                if 'footprint' in registro.dtype.names:
                    campo_intermedio = registro['footprint']
                    nombre_campo_final = 'img'
                elif 'gait' in registro.dtype.names:
                    campo_intermedio = registro['gait']
                    nombre_campo_final = 'gait'

                # Si se encontró la estructura esperada, procesar la imagen
                if campo_intermedio is not None and campo_intermedio.size > 0:
                    sub_estructura = campo_intermedio.flat[0]
                    if nombre_campo_final in sub_estructura.dtype.names:
                        imagen_data = sub_estructura[nombre_campo_final].astype(np.float64)

                        if guardar_todas:
                            nombre_archivo = f"sujeto_{id_sujeto}_registro_{i}.png"
                            ruta_guardado = p_salida / nombre_archivo
                            plt.imsave(ruta_guardado, imagen_data, cmap='gray')
                            contador_guardadas += 1
                        else:
                            # Modo "mostrar primera imagen"
                            print(f"Mostrando imagen del registro {i} (ID de sujeto: {id_sujeto})")
                            plt.imshow(imagen_data, cmap='gray')
                            plt.title(f"Vista previa de '{p_archivo.name}'\nSujeto {id_sujeto}, Registro {i}")
                            plt.axis('off')
                            plt.show()
                            # Terminamos después de mostrar la primera
                            return

            except (AttributeError, KeyError, TypeError) as e:
                # Si un registro no tiene la estructura esperada, lo saltamos
                print(f"[WARN] Omitiendo registro {i} por estructura inválida o campo faltante: {e}")
                continue

        if guardar_todas:
            print(f"\n[ÉXITO] Proceso completado. Se guardaron {contador_guardadas} imágenes en '{p_salida}'.")

    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado al procesar el archivo: {e}")

