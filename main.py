#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación Principal - Sistema de Análisis de Presiones Plantares

DESCRIPCIÓN:
    Sistema interactivo completo para el procesamiento y análisis de datos de
    presión plantar provenientes de tres datasets diferentes: CASIA-D, CADDataset
    (Footscan) y StepUpDataset (StepUp-P150).
    
    La aplicación proporciona un menú interactivo con 7 opciones que cubren todo
    el flujo de trabajo: desde la extracción de imágenes hasta el cálculo de
    parámetros biomecánicos avanzados.

DATASETS SOPORTADOS:

    1. CASIA-D (Chinese Academy of Sciences):
       - Formato: Archivos .mat de MATLAB
       - Contenido: Huellas estáticas y patrones de marcha
       - Uso: Investigación en reconocimiento biométrico
    
    2. CADDataset (Footscan):
       - Formato: Archivos .nii (NIfTI) con secuencias temporales
       - Frecuencia: 500 fps (pacientes C) o 200 Hz (voluntarios HV)
       - Resolución: 7.62 mm × 5.08 mm por píxel (píxeles rectangulares)
       - Uso: Análisis clínico de marcha
    
    3. StepUpDataset (StepUp-P150):
       - Formato: Archivos .npz con trials completos
       - Frecuencia: 100 Hz
       - Resolución: 0.5 cm × 0.5 cm por píxel (píxeles cuadrados)
       - Uso: Investigación en biomecánica de la marcha

MENÚ DE OPCIONES:

    1. EXTRAER IMÁGENES DESDE .MAT (CASIA)
       - Procesa archivos CASfootprint.mat o CASgait.mat
       - Extrae todas las imágenes como PNG individuales
       - Módulo: Download_mat.py
    
    2. EXTRAER IMÁGENES PEAK DESDE .NII (CAD)
       - Procesa directorios con archivos .nii
       - Genera imágenes de presión máxima (peak)
       - Opcionalmente crea GIFs animados
       - Módulo: Download_nii.py
    
    3. EXTRAER IMÁGENES PEAK DE STEPUP (.NPZ)
       - Procesa datos del StepUpDataset
       - Exporta imágenes peak de pisadas específicas
       - Filtra pisadas válidas automáticamente
       - Módulos: StepUpDataset/utils.py
    
    4. ANALIZAR UNA IMAGEN INDIVIDUAL
       - Endereza y normaliza una imagen plantar
       - Calcula el Centro de Presión (CoP)
       - Calcula dimensiones reales en centímetros
       - Identifica la región del pie donde cae el CoP
       - Muestra visualización con máscaras de regiones
       - Módulos: Rotate.py, Standarization.py, Compute_cop_static.py, Pixel_to_cm.py
    
    5. CALCULAR PARÁMETROS PARA PACIENTE STEPUP
       - Calcula CoF (distribución de fuerza) y CoP por región
       - Genera CSVs: Footsteps_detail.csv y Summary.csv
       - Opcionalmente genera GIF de trayectoria CoP
       - Módulos: Parameters_StepUp.py, GIF_Generator.py
    
    6. CALCULAR PARÁMETROS PARA PACIENTE CAD
       - Calcula CoF y CoP para todos los trials de un paciente
       - Detecta automáticamente tipo de paciente (C o HV) y ajusta frecuencia
       - Genera CSVs con resumen por lado (Left, Right, Overall)
       - Opcionalmente genera GIF de trayectoria CoP
       - Módulos: Parameters_CAD.py, GIF_Generator.py
    
    7. SALIR DEL PROGRAMA
       - Cierra la aplicación de forma ordenada

CARACTERÍSTICAS PRINCIPALES:

    - Interfaz de usuario amigable con menús interactivos
    - Diálogos de selección de archivos (tkinter)
    - Validación de entradas del usuario
    - Manejo robusto de errores con mensajes claros
    - Opción de continuar o salir después de cada operación
    - Visualizaciones interactivas con matplotlib
    - Soporte para múltiples datasets sin cambiar código

PARÁMETROS GLOBALES:
    - DEFAULT_RATIOS: (0.30, 0.55, 0.85) - División del pie en regiones
    - FOOTWEAR_LIST: ["BF", "ST", "P1", "P2"] - Tipos de calzado StepUp
    - WALK_CONDITIONS: ["W1", "W2", "W3", "W4"] - Condiciones de marcha StepUp

FLUJO DE TRABAJO TÍPICO:

    Para análisis de CAD:
        1. Opción 2: Extraer imágenes peak desde .nii
        2. Opción 4: Analizar imagen individual (exploración)
        3. Opción 6: Calcular parámetros completos + GIF
    
    Para análisis de StepUp:
        1. Opción 3: Extraer imágenes peak (opcional)
        2. Opción 5: Calcular parámetros + GIF

FORMATO DE SALIDA:
    - CSVs: Separador ';', decimales con ',' (formato europeo)
    - Imágenes: PNG con DPI=150
    - GIFs: 20 fps, colormap jet, trayectoria CoP en verde

REQUISITOS:
    - Python 3.9+
    - Librerías: numpy, matplotlib, opencv-python, pillow, nibabel, scipy, pandas

AUTOR: Valentina López
PROYECTO: TFG - Análisis de Presiones Plantares
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
from PIL import Image

# Módulos del proyecto
from Download_mat import procesar_casia_mat
from Download_nii import descargar_img
from Standarization import normalize_image
from Rotate import enderezar_por_contorno
from Compute_cop_static import compute_cop_static
from Pixel_to_cm import calculate_foot_dimensions_from_pressure

# Parámetros para diferentes datasets
from Parameters_StepUp import export_parameters as export_parameters_stepup
from Parameters_StepUp import DEFAULT_RATIOS, _compute_orientation, _region_masks, _cop_region
from Parameters_CAD import export_parameters_from_directory as export_parameters_cad

# StepUpDataset utilities
try:
    from StepUpDataset.utils import load_metadata, load_footsteps, dataset_folder, load_trial
except Exception:
    load_metadata = None  # type: ignore
    load_footsteps = None  # type: ignore
    dataset_folder = None  # type: ignore
    load_trial = None  # type: ignore


# ============================================================================
# UTILIDADES GENERALES
# ============================================================================


def _to_u8_minmax(x: np.ndarray) -> np.ndarray:
    """
    Normaliza un array al rango 0-255 usando normalización min-max.
    
    Útil para convertir datos de presión a formato de imagen visualizable.
    
    Args:
        x: Array numpy con valores de presión (cualquier rango)
    
    Returns:
        Array uint8 con valores normalizados en rango [0, 255]
    
    Note:
        - Si el array está vacío o tiene valores inválidos, retorna array de ceros
        - Maneja NaN y valores infinitos de forma segura
    """
    x = np.asarray(x, dtype=np.float32)
    m = float(np.nanmin(x))
    M = float(np.nanmax(x))
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - m) / (M - m)
    return (y * 255.0).clip(0, 255).astype(np.uint8)


def _jet_with_zero_black() -> mpl.colors.ListedColormap:
    """
    Crea un colormap jet modificado donde el valor 0 se muestra como negro.
    
    El colormap jet estándar muestra 0 como azul oscuro, pero para visualización
    de presiones plantares es mejor mostrar 0 (sin presión) como negro puro.
    
    Returns:
        ListedColormap: Colormap jet con primer color modificado a negro
    
    Note:
        - Usado en la Opción 3 para exportar imágenes peak de StepUp
        - Mejora la visualización al distinguir claramente áreas sin presión
    """
    jet = mpl.colormaps["jet"](np.linspace(0, 1, 256))
    jet[0] = [0, 0, 0, 1]
    return mpl.colors.ListedColormap(jet)


def _colorize_jet_u8(gray_u8: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen en escala de grises a RGB usando colormap jet.
    
    Args:
        gray_u8: Array numpy uint8 con valores en rango [0, 255]
    
    Returns:
        Array numpy uint8 RGB (H, W, 3) con colores del jet colormap
    
    Note:
        - Usa _jet_with_zero_black() para que 0 sea negro
        - Azul = baja presión, Rojo = alta presión
    """
    cm = _jet_with_zero_black()
    rgb = cm(gray_u8)[..., :3]
    return (rgb * 255).astype(np.uint8)


# ============================================================================
# OPCIÓN 1: EXTRAER IMÁGENES DESDE .MAT (CASIA)
# ============================================================================


def opcion_extraer_mat():
    """
    OPCIÓN 1: Extrae imágenes desde archivo .mat del dataset CASIA-D.
    
    Proceso interactivo:
        1. Usuario selecciona archivo .mat (CASfootprint.mat o CASgait.mat)
        2. Usuario selecciona carpeta de destino
        3. El sistema extrae todas las imágenes automáticamente
        4. Guarda cada imagen como: sujeto_{id}_registro_{num}.png
    
    Módulo utilizado:
        - Download_mat.py: procesar_casia_mat()
    
    Salida típica:
        - Decenas o cientos de imágenes PNG (una por registro)
        - Nombres descriptivos con ID de sujeto y número de registro
    """
    print("\n" + "=" * 60)
    print("EXTRAER IMÁGENES DESDE .MAT (CASIA)")
    print("=" * 60)

    print("\nSelecciona el archivo .mat que quieres procesar...")
    ruta_archivo = filedialog.askopenfilename(
        title="Selecciona archivo .mat",
        filetypes=[("MAT files", "*.mat")]
    )
    if not ruta_archivo:
        print("❌ Acción cancelada. Volviendo al menú...")
        return

    print("\nSelecciona la carpeta donde guardar las imágenes...")
    carpeta_salida = filedialog.askdirectory(
        title="Selecciona carpeta de destino"
    )
    if not carpeta_salida:
        print("❌ Acción cancelada. Volviendo al menú...")
        return

    try:
        procesar_casia_mat(ruta_archivo, carpeta_salida)
        print("\n✅ Proceso completado exitosamente!")
    except Exception as e:
        print(f"\n❌ Error al procesar: {e}")


# ============================================================================
# OPCIÓN 2: EXTRAER IMÁGENES PEAK DESDE CARPETA .NII (CAD)
# ============================================================================


def opcion_extraer_nii():
    """
    OPCIÓN 2: Extrae imágenes peak desde carpeta con archivos .nii del CADDataset.
    
    Proceso interactivo:
        1. Usuario selecciona carpeta con archivos .nii (ej: CADDataset/C01/)
        2. Usuario selecciona carpeta de destino
        3. El sistema procesa todos los .nii encontrados recursivamente
        4. Guarda imagen de presión máxima (peak) de cada trial
    
    Configuración:
        - export_peak=True: Guarda imagen PNG del peak
        - export_gif=False: No genera GIFs (se pueden generar en Opción 6)
        - rotate=False: No rota (el enderezamiento se hace en análisis posterior)
    
    Módulo utilizado:
        - Download_nii.py: descargar_img()
    
    Salida típica:
        - Una imagen PNG por cada archivo .nii
        - Mantiene estructura de carpetas original
        - Ejemplo: left_foot_trial_01.nii → left_foot_trial_01.png
    """
    print("\n" + "=" * 60)
    print("EXTRAER IMÁGENES PEAK DESDE .NII (CAD)")
    print("=" * 60)

    print("\nSelecciona la CARPETA que contiene tus archivos .nii...")
    ruta_carpeta = filedialog.askdirectory(
        title="Selecciona carpeta con archivos .nii"
    )
    if not ruta_carpeta:
        print("❌ Acción cancelada. Volviendo al menú...")
        return

    print("\nSelecciona la carpeta donde guardar las imágenes...")
    carpeta_salida = filedialog.askdirectory(
        title="Selecciona carpeta de destino"
    )
    if not carpeta_salida:
        print("❌ Acción cancelada. Volviendo al menú...")
        return

    try:
        descargar_img(
            ruta_carpeta,
            carpeta_salida,
            export_peak=True,
            export_gif=False,
            rotate=False
        )
        print("\n✅ Proceso de extracción completado exitosamente!")
    except Exception as e:
        print(f"\n❌ Error al procesar: {e}")


# ============================================================================
# OPCIÓN 3: EXTRAER IMÁGENES PEAK DE STEPUP (.NPZ)
# ============================================================================


FOOTWEAR_LIST: List[str] = ["BF", "ST", "P1", "P2"]
WALK_CONDITIONS: List[str] = ["W1", "W2", "W3", "W4"]


def _make_outdir(base: Path, participant_id: int, footwear: str, walk: str) -> Path:
    """
    Crea la estructura de directorios para exportar imágenes de StepUp.
    
    Args:
        base: Directorio base de salida
        participant_id: ID del participante (número entero)
        footwear: Tipo de calzado ("BF", "ST", "P1", "P2")
        walk: Condición de marcha ("W1", "W2", "W3", "W4")
    
    Returns:
        Path: Ruta al directorio creado
    
    Estructura creada:
        base/{participant_id:03d}/{footwear}/{walk}/
        Ejemplo: output/001/BF/W1/
    """
    d = base / f"{participant_id:03d}" / footwear / walk
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_peaks_for_participant(
    out_root: Path,
    participant_id: int,
    footwear_list: List[str],
    walk_list: List[str],
) -> None:
    """
    Exporta imágenes de presión máxima (peak) para un participante de StepUp.
    
    Procesa todas las combinaciones de footwear y walk condition especificadas,
    cargando los datos del pipeline_1 (datos procesados) y generando imágenes
    PNG coloreadas con jet colormap.
    
    Args:
        out_root: Directorio raíz de salida
        participant_id: ID del participante (número entero)
        footwear_list: Lista de tipos de calzado a procesar (ej: ["BF", "ST"])
        walk_list: Lista de condiciones de marcha a procesar (ej: ["W1", "W2"])
    
    Proceso:
        1. Para cada combinación footwear/walk:
           a. Carga metadata del participante
           b. Carga footsteps del pipeline_1.npz
           c. Filtra pisadas excluidas según metadata["Exclude"]
           d. Para cada pisada válida:
              - Calcula presión máxima: footsteps[i].max(axis=0)
              - Normaliza a uint8 [0, 255]
              - Colorea con jet colormap (0=negro)
              - Guarda como PNG: step_{i:03d}_peak.png
        2. Reporta progreso por cada combinación
        3. Muestra total de imágenes exportadas
    
    Filtrado de pisadas:
        - Si existe columna "Exclude" en metadata: excluye pisadas marcadas
        - Si no existe: incluye todas las pisadas
    
    Manejo de errores:
        - Combinaciones sin datos se omiten con warning
        - Formato inesperado de footsteps se omite
        - Errores no detienen el proceso completo
    
    Note:
        - Usa pipeline_1 (datos ya procesados y normalizados)
        - El colormap jet con 0=negro mejora visualización
        - Los nombres de archivo incluyen índice con 3 dígitos (000, 001, etc.)
    """
    if load_metadata is None or load_footsteps is None:
        messagebox.showerror(
            "StepUpDataset",
            "No se pudo importar StepUpDataset.utils.\n"
            "Revisa el import o la ruta del proyecto."
        )
        return

    total_exported = 0
    for footwear in footwear_list:
        for walk in walk_list:
            try:
                metadata = load_metadata(participant_id, footwear, walk)
            except Exception:
                print(f"⚠️  Sin metadata para {participant_id:03d}/{footwear}/{walk}")
                continue

            try:
                footsteps = load_footsteps(participant_id, footwear, walk, pipeline=1)
            except Exception:
                print(f"⚠️  Sin pipeline_1.npz para {participant_id:03d}/{footwear}/{walk}")
                continue

            if footsteps.ndim != 4:
                print(f"⚠️  Formato inesperado {footsteps.shape}, salto.")
                continue

            n_steps = footsteps.shape[0]
            if "Exclude" in metadata.columns and len(metadata) >= n_steps:
                keep = ~metadata["Exclude"].astype(bool).values[:n_steps]
            else:
                keep = np.ones(n_steps, dtype=bool)

            out_dir = _make_outdir(out_root, participant_id, footwear, walk)
            exported_here = 0

            for i in range(n_steps):
                if not keep[i]:
                    continue
                img_peak = footsteps[i].max(axis=0)  # (H, W)
                gray_u8 = _to_u8_minmax(img_peak)
                rgb_u8 = _colorize_jet_u8(gray_u8)
                plt.imsave(out_dir / f"step_{i:03d}_peak.png", rgb_u8)
                exported_here += 1
                total_exported += 1

            print(f"✓ {participant_id:03d}/{footwear}/{walk}: {exported_here} imágenes -> {out_dir}")

    print(f"\n✅ Exportadas {total_exported} imágenes peak (participante {participant_id:03d})")


def opcion_extraer_stepup():
    """
    OPCIÓN 3: Extrae imágenes peak del StepUpDataset (.npz).
    
    Proceso interactivo:
        1. Usuario ingresa participant_id (número entero)
        2. Usuario selecciona carpeta de destino
        3. El sistema carga datos del participante (BF/W1 por defecto)
        4. Exporta imágenes peak de todas las pisadas válidas
        5. Organiza en subcarpetas: {participant_id}/BF/W1/
    
    Filtrado automático:
        - Solo pisadas válidas (Exclude=False en metadata)
        - Ignora pisadas marcadas como excluidas
    
    Configuración actual:
        - Footwear: "BF" (Barefoot - descalzo)
        - Walk condition: "W1" (condición de marcha 1)
        - Colormap: Jet con 0=negro
    
    Módulos utilizados:
        - StepUpDataset/utils.py: load_metadata(), load_footsteps()
        - Funciones internas: export_peaks_for_participant()
    
    Salida típica:
        - Múltiples PNG organizados por participante/footwear/walk
        - Nombres: step_000_peak.png, step_001_peak.png, etc.
        - Directorio: {output}/001/BF/W1/
    
    Note:
        - Requiere que StepUpDataset/utils.py esté correctamente configurado
        - dataset_folder debe apuntar a la ubicación correcta del dataset
    """
    print("\n" + "=" * 60)
    print("EXTRAER IMÁGENES PEAK DE STEPUP (.NPZ)")
    print("=" * 60)

    # Validar imports
    if load_metadata is None or load_footsteps is None:
        messagebox.showerror(
            "StepUpDataset",
            "No se pudo importar StepUpDataset.utils.\n"
            "Asegúrate de tener StepUpDataset/utils.py correctamente configurado."
        )
        return

    # Verificar dataset_folder
    try:
        ds_path = Path(dataset_folder)  # type: ignore
        if not ds_path.exists():
            messagebox.showwarning(
                "StepUpDataset",
                f"La carpeta del dataset no existe: {dataset_folder}\n"
                "Edita StepUpDataset/utils.py para configurar la ruta correcta."
            )
            return
    except Exception:
        messagebox.showerror("StepUpDataset", "Error al acceder a dataset_folder.")
        return

    # Pedir participant_id
    participant_text = input(
        "\nIngresa el participant_id (ej. 1, 2, 3): "
    ).strip()
    if not participant_text.isdigit():
        print("❌ participant_id inválido. Debe ser un número entero.")
        return

    participant_id = int(participant_text)

    # Seleccionar carpeta de salida
    messagebox.showinfo(
        "Exportar StepUp",
        "Selecciona la CARPETA de destino para las imágenes PNG."
    )
    out_dir = filedialog.askdirectory(title="Selecciona carpeta de salida")
    if not out_dir:
        print("❌ Acción cancelada. Volviendo al menú...")
        return

    try:
        export_peaks_for_participant(
            Path(out_dir),
            participant_id,
            ["BF"],
            ["W1"]
        )
        print("\n✅ Proceso completado exitosamente!")
    except Exception as e:
        print(f"\n❌ Error al exportar: {e}")


# ============================================================================
# OPCIÓN 4: ANALIZAR UNA IMAGEN INDIVIDUAL
# ============================================================================


def opcion_analizar_imagen():
    """
    OPCIÓN 4: Analiza una imagen plantar individual de forma completa.
    
    Pipeline de análisis:
        1. Usuario selecciona imagen (PNG, JPG, etc.)
        2. Usuario indica dataset de origen (CAD o CASIA)
        3. Enderezamiento:
           - Detecta contorno del pie
           - Calcula ángulo de inclinación usando PCA
           - Rota imagen para alinear eje longitudinal vertical
           - Muestra proceso en 6 imágenes (si mostrar_plots=True)
        4. Normalización:
           - Ajusta brillo (factor 1.2)
           - Ajusta contraste (factor 1.3)
           - Mantiene tamaño original (output_size=None)
        5. Cálculo de CoP:
           - Calcula centro de presión de la imagen enderezada
           - Identifica en qué región del pie cae (retropié/mediopié/antepié/dedos)
        6. Cálculo de dimensiones reales:
           - Convierte píxeles a centímetros según dataset
           - Calcula ancho, alto y área del pie en cm
        7. Visualización final:
           - Imagen original vs imagen procesada
           - CoP marcado con punto rojo
           - Líneas verdes de división de regiones
           - Información de región del CoP
    
    Módulos utilizados:
        - Rotate.py: enderezar_por_contorno()
        - Standarization.py: normalize_image()
        - Compute_cop_static.py: compute_cop_static()
        - Pixel_to_cm.py: calculate_foot_dimensions_from_pressure()
        - Parameters_StepUp.py: _compute_orientation(), _region_masks(), _cop_region()
    
    Configuraciones por dataset:
        - CAD: Kernel cierre 25×25, sin apertura, grosor 5px
        - CASIA: Kernel cierre 6×6, apertura 5×5, grosor 1px
    
    Salida:
        - Visualización interactiva con matplotlib (2 paneles)
        - Información impresa: CoP, dimensiones, región
        - No guarda archivos (solo visualización)
    
    Note:
        - Las líneas verdes divisorias son dibujadas por Rotate.py
        - El CoP se calcula sobre la imagen YA enderezada
        - La conversión a cm usa escalas conocidas por dataset
    """
    print("\n" + "=" * 60)
    print("ANALIZAR IMAGEN INDIVIDUAL")
    print("=" * 60)

    print("\nSelecciona la imagen que quieres analizar...")
    ruta_imagen = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )
    if not ruta_imagen:
        print("❌ Acción cancelada. Volviendo al menú...")
        return

    # Preguntar de qué dataset proviene
    print("\n¿De qué dataset proviene esta imagen?")
    print("  1. CAD")
    print("  2. CASIA")
    
    try:
        opcion_dataset = int(input("\nElige una opción (1-2): ").strip())
        dataset_map = {1: 'cad', 2: 'casia'}
        dataset_origen = dataset_map.get(opcion_dataset)
        if opcion_dataset not in dataset_map:
            print("⚠️  Opción inválida, usando CAD por defecto...")
            dataset_origen = 'cad'
    except (ValueError, KeyboardInterrupt):
        print("⚠️  Entrada inválida, usando CAD por defecto...")
        dataset_origen = 'cad'

    print(f"\n📊 Analizando: {Path(ruta_imagen).name}")

    try:
        # Determinar configuración de rotación según dataset
        if dataset_origen == 'casia':
            nombre_db = 'CASIA'
        else:  # cad
            nombre_db = 'CAD'
        
        print(f"\n🔄 Aplicando rotación con configuración {nombre_db}...")
        print("=" * 60)
        
        # Llamar a enderezar_por_contorno con visualización
        imagen_enderezada, angulo = enderezar_por_contorno(
            ruta_imagen, 
            nombre_db=nombre_db, 
            mostrar_plots=True  # Mostrar las 6 imágenes del proceso
        )
        
        print("=" * 60)
        print(f"✓ Imagen enderezada con corrección de {angulo:.2f}°")
        print("\n💡 Cierra la ventana de visualización de rotación para continuar...")
        
        # Esperar a que el usuario cierre la ventana de rotación
        # (plt.show() en enderezar_por_contorno es bloqueante)

        print("\n📈 Normalizando imagen enderezada...")
        
        # Convertir de numpy array (OpenCV BGR) a PIL Image (RGB)
        if isinstance(imagen_enderezada, np.ndarray):
            # OpenCV usa BGR, PIL usa RGB
            imagen_rgb = cv2.cvtColor(imagen_enderezada, cv2.COLOR_BGR2RGB)
            imagen_pil = Image.fromarray(imagen_rgb)
        else:
            # Si ya es PIL Image, usar directamente
            imagen_pil = imagen_enderezada

        processed_img = normalize_image(
            imagen_pil,
            output_size=None,
            brightness_factor=1.2,
            contrast_factor=1.3
        )

        # Convertir imagen procesada a array numpy para calcular CoP
        if isinstance(processed_img, Image.Image):
            img_array = np.array(processed_img.convert('L'), dtype=np.float64)
        else:
            img_array = np.asarray(processed_img, dtype=np.float64)
            # Si es RGB, convertir a escala de grises
            if img_array.ndim == 3:
                img_array = np.mean(img_array, axis=2)

        # Calcular CoP de la imagen procesada (ya enderezada)
        print("\n📍 Calculando Centro de Presión (CoP)...")
        cop_x, cop_y = compute_cop_static(img_array)
        
        if np.isfinite(cop_x) and np.isfinite(cop_y):
            print(f"✓ CoP calculado: ({cop_x:.2f}, {cop_y:.2f}) píxeles")
            print(f"  CoP relativo: ({cop_x/img_array.shape[1]:.2%}, {cop_y/img_array.shape[0]:.2%})")
        else:
            print("⚠️  No se pudo calcular el CoP (imagen sin presión válida)")
        
        # Calcular dimensiones reales del pie en centímetros
        if dataset_origen:
            dims = calculate_foot_dimensions_from_pressure(img_array, dataset=dataset_origen)
            if 'error' in dims:
                print(f"⚠️  {dims['error']}")
            elif dims['width_cm'] is not None:
                print(f"✓ Dimensiones del pie: {dims['width_cm']:.2f} cm × {dims['height_cm']:.2f} cm")
                print(f"  Área del pie: {dims['area_cm2']:.2f} cm²")
                print(f"  Dataset: {dataset_origen.upper()}")
            else:
                print(f"ℹ️  Información de escala no disponible para {dataset_origen.upper()}")
                print(f"  Dimensiones en píxeles: {dims['width_px']:.2f} px × {dims['height_px']:.2f} px")
        else:
            print("ℹ️  Sin información de dataset - dimensiones no calculadas")

        # Calcular orientación y máscaras de regiones
        is_horizontal, invert = _compute_orientation(img_array)
        masks, norm_img = _region_masks(
            img_array.shape, is_horizontal, invert, DEFAULT_RATIOS
        )
        
        # Determinar región del CoP si es válido
        cop_region = None
        if np.isfinite(cop_x) and np.isfinite(cop_y):
            cop_region = _cop_region(cop_x, cop_y, norm_img, is_horizontal, DEFAULT_RATIOS)
        
        # Las líneas divisorias ya fueron dibujadas en la imagen por Rotate.py
        # No es necesario calcularlas aquí

        # Crear visualización
        print("\n🎯 Calculando y visualizando CoP sobre imagen enderezada...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Imagen original
        axes[0].set_title("Imagen Original", fontsize=12, fontweight="bold")
        axes[0].imshow(Image.open(ruta_imagen))
        axes[0].axis("off")

        # Imagen procesada con CoP marcado y máscaras de regiones
        if np.isfinite(cop_x) and np.isfinite(cop_y):
            titulo = f"Imagen Enderezada + CoP\nCoP: ({cop_x:.1f}, {cop_y:.1f}) px"
            if cop_region:
                titulo += f" - Región: {cop_region}"
        else:
            titulo = "Imagen Enderezada + Regiones"
        
        axes[1].set_title(titulo, fontsize=12, fontweight="bold")
        axes[1].imshow(processed_img)
        
        # Las líneas divisorias verdes ya están dibujadas en la imagen por Rotate.py
        # No dibujar líneas adicionales aquí
        
        # Marcar el CoP si es válido
        if np.isfinite(cop_x) and np.isfinite(cop_y):
            axes[1].plot(
                cop_x,
                cop_y,
                "ro",
                markersize=12,
                markeredgecolor="yellow",
                markeredgewidth=2,
                label=f"CoP: ({cop_x:.1f}, {cop_y:.1f})",
                zorder=5,
            )
            axes[1].legend(loc="best", fontsize=10)
        
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        print("\n✅ Análisis completado!")

    except Exception as e:
        print(f"\n❌ Error al analizar la imagen: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# OPCIÓN 5: CALCULAR PARÁMETROS PARA PACIENTE STEPUP
# ============================================================================


def opcion_parametros_stepup():
    """
    OPCIÓN 5: Calcula parámetros biomecánicos completos para un participante de StepUp.
    
    Proceso interactivo:
        1. Usuario ingresa participant_id (número entero)
        2. El sistema carga todos los datos del participante (BF/W1)
        3. Filtra pisadas válidas (Standing=0, Incomplete=0)
        4. Calcula CoF y CoP para cada pisada
        5. Genera dos CSVs:
           - Footsteps_detail.csv: Detalle de cada pisada
           - Summary.csv: Resumen por lado (Left, Right, Overall)
        6. Opcionalmente genera GIF de una pisada específica:
           - Muestra lista de pisadas disponibles
           - Usuario selecciona una pisada
           - Genera GIF animado con trayectoria CoP
    
    Configuración fija:
        - Shoe type: "BF" (Barefoot - descalzo)
        - Walk condition: "W1" (velocidad normal)
        - FPS: 100 Hz (obtenido del metadata o fallback)
        - Ratios: DEFAULT_RATIOS (0.30, 0.55, 0.85)
    
    Módulos utilizados:
        - Parameters_StepUp.py: export_parameters()
        - GIF_Generator.py: create_gif_stepup() (opcional)
        - StepUpDataset/utils.py: load_metadata(), load_trial()
    
    Salida:
        - Directorio: Salida_StepUp/Participante_{id}/BF/W1/
        - Archivos: Footsteps_detail.csv, Summary.csv
        - GIF (opcional): cop_trajectory_pisada{N}_pass{P}_foot{F}.gif
    
    Métricas en CSV:
        - Stance time (s): Duración de la fase de apoyo
        - Num. frames valid: Frames con presión > 0
        - Mean CoF por región (%): Distribución promedio de fuerza
        - Frames CoP por región (%): Porcentaje de tiempo del CoP en cada región
    
    Note:
        - Los frames se enderezan automáticamente antes de calcular métricas
        - Se aplica inversión física cuando es necesario (talón abajo)
        - El GIF muestra la evolución temporal del CoP con líneas de región
    """
    print("\n" + "=" * 60)
    print("CALCULAR PARÁMETROS - STEPUP")
    print("=" * 60)

    participant_text = input(
        "\nIngresa el participant_id (ej. 1, 2, 3): "
    ).strip()
    if not participant_text.isdigit():
        print("❌ participant_id inválido. Debe ser un número entero.")
        return

    participant_id = int(participant_text)
    print(f"\n📊 Calculando parámetros para participante {participant_id:03d}...")

    base_out_dir = Path("Salida_StepUp") / f"Participante_{participant_id:03d}" / "BF" / "W1"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        detail_path, summary_path = export_params_stepup_cli(participant_id, base_out_dir)

        print(f"\n✅ CSV generados para participante {participant_id:03d}/BF/W1:")
        print(f"   📄 Detalle: {detail_path}")
        print(f"   📄 Resumen: {summary_path}")

        manejar_visualizaciones_stepup(participant_id, base_out_dir)

    except Exception as e:
        print(f"\n❌ Error al generar parámetros: {e}")


def export_params_stepup_cli(participant_id: int, base_out_dir: Path):
    """
    Wrapper para llamar a export_parameters de StepUp con configuración CLI.
    
    Simplifica la llamada desde opcion_parametros_stepup() usando valores
    predeterminados para shoe_type y walk_condition.
    
    Args:
        participant_id: ID del participante
        base_out_dir: Directorio de salida
    
    Returns:
        Tuple[Path, Path]: Rutas a los CSVs generados (detail, summary)
    """
    return export_parameters_stepup(
        participant_id=participant_id,
        shoe_type="BF",
        walk_condition="W1",
        output_dir=base_out_dir,
        fps_fallback=100.0,
        ratios=DEFAULT_RATIOS,
    )


def manejar_visualizaciones_stepup(participant_id: int, base_out_dir: Path):
    """
    Maneja la generación opcional de GIFs para StepUp después de calcular parámetros.
    
    Proceso interactivo:
        1. Pregunta al usuario si desea generar un GIF
        2. Si acepta:
           a. Carga metadata y filtra pisadas válidas
           b. Muestra lista de pisadas disponibles
           c. Usuario selecciona una pisada por número
           d. Genera GIF de la trayectoria CoP de esa pisada
    
    Args:
        participant_id: ID del participante
        base_out_dir: Directorio donde guardar el GIF
    
    Información mostrada:
        - Lista de pisadas con: [índice] Pass | Footstep | Lado
        - Si hay más de 15 pisadas: muestra primeras 10 y últimas 5
    
    GIF generado:
        - Nombre: cop_trajectory_pisada{N}_pass{P}_foot{F}.gif
        - Contenido: Animación de presión + trayectoria CoP
        - Configuración: 20 fps, ratios DEFAULT_RATIOS
    
    Note:
        - Solo procesa pisadas válidas (Standing=0, Incomplete=0)
        - Requiere matplotlib instalado
        - El GIF se guarda en el mismo directorio que los CSVs
    """
    print("\n" + "-" * 60)
    generar_gif = input("¿Desea generar un GIF de la trayectoria CoF? (s/n): ").strip().lower()

    if generar_gif not in ['s', 'si', 'sí', 'yes', 'y']:
        return

    if load_metadata is None:
        print("❌ No se pudo cargar StepUpDataset.utils")
        return

    metadata = load_metadata(participant_id, "BF", "W1")
    mask = (metadata["Standing"] == 0) & (metadata["Incomplete"] == 0)
    steps = metadata[mask].sort_values(["PassID", "FootstepID"])

    if steps.empty:
        print("❌ No hay pisadas válidas para este participante.")
        return

    mostrar_lista_pisadas(steps)

    pisada_idx = seleccionar_indice(len(steps))
    if pisada_idx is None:
        return

    selected_step = steps.iloc[pisada_idx - 1]
    pass_id = int(selected_step['PassID'])
    footstep_id = int(selected_step['FootstepID'])
    side = selected_step['Side']

    print(f"\n✓ Seleccionada: Pisada {pisada_idx} → Pass {pass_id}, "
          f"Footstep {footstep_id} ({side})")

    try:
        from GIF_Generator import create_gif_stepup

        gif_path = base_out_dir / f"cop_trajectory_pisada{pisada_idx}_pass{pass_id}_foot{footstep_id}.gif"
        create_gif_stepup(
            participant_id=participant_id,
            shoe_type="BF",
            walk_condition="W1",
            pass_id=pass_id,
            footstep_id=footstep_id,
            output_path=gif_path,
            ratios=DEFAULT_RATIOS,
            fps_gif=20,
        )
        print(f"\n✅ GIF generado exitosamente!")
        print(f"   🎬 Ubicación: {gif_path}")
        print(f"   📏 Tamaño: {gif_path.stat().st_size / 1024:.1f} KB")
    except ImportError:
        print("\n⚠️  No se pudo generar el GIF: matplotlib no está instalado.")
        print("   Instala con: pip install matplotlib")
    except Exception as e:
        print(f"\n❌ Error al generar GIF: {e}")


def mostrar_lista_pisadas(steps):
    """
    Muestra una lista formateada de pisadas disponibles para StepUp.
    
    Formatea y muestra la información de pisadas de forma legible, con manejo
    inteligente de listas largas (muestra primeras y últimas si hay muchas).
    
    Args:
        steps: DataFrame de pandas con pisadas filtradas (Standing=0, Incomplete=0)
               Debe contener columnas: PassID, FootstepID, Side
    
    Formato de salida:
        [índice] Pass XX | Footstep XXX | Lado: Left/Right
    
    Lógica de display:
        - Si <= 15 pisadas: Muestra todas
        - Si > 15 pisadas: Muestra primeras 10 + últimas 5
    
    Note:
        - Los índices empiezan en 1 (no en 0) para facilitar selección del usuario
        - Formato alineado para mejor legibilidad
    """
    print(f"\n📋 Pisadas disponibles ({len(steps)} en total):")
    print("-" * 60)
    display_limit = 10
    if len(steps) <= 15:
        for idx, (_, step) in enumerate(steps.iterrows(), 1):
            print(f"  [{idx:3d}] Pass {int(step['PassID']):2d} | "
                  f"Footstep {int(step['FootstepID']):3d} | "
                  f"Lado: {step['Side']}")
    else:
        for idx, (_, step) in enumerate(steps.head(display_limit).iterrows(), 1):
            print(f"  [{idx:3d}] Pass {int(step['PassID']):2d} | "
                  f"Footstep {int(step['FootstepID']):3d} | "
                  f"Lado: {step['Side']}")
        print("  ...")
        print(f"  (Mostrando 10 de {len(steps)} pisadas)")
        print("\n  Últimas pisadas:")
        start_idx = len(steps) - 5
        for idx, (_, step) in enumerate(steps.tail(5).iterrows(), start_idx + 1):
            print(f"  [{idx:3d}] Pass {int(step['PassID']):2d} | "
                  f"Footstep {int(step['FootstepID']):3d} | "
                  f"Lado: {step['Side']}")


def seleccionar_indice(max_items: int) -> int | None:
    """
    Solicita al usuario que seleccione un índice de una lista y valida la entrada.
    
    Args:
        max_items: Número máximo de items disponibles
    
    Returns:
        int | None: Índice seleccionado (1-based) o None si la entrada es inválida
    
    Validaciones:
        - Verifica que la entrada sea un número entero
        - Verifica que esté en el rango [1, max_items]
        - Muestra mensajes de error claros si la validación falla
    
    Note:
        - Los índices son 1-based para facilitar la interacción con el usuario
        - Retorna None en caso de error (el llamador debe manejar esto)
    """
    print("-" * 60)
    pisada_text = input(f"\nIngresa el número de pisada (1-{max_items}): ").strip()
    if not pisada_text.isdigit():
        print("❌ Número inválido.")
        return None

    pisada_idx = int(pisada_text)
    if not (1 <= pisada_idx <= max_items):
        print(f"❌ Número fuera de rango. Debe estar entre 1 y {max_items}.")
        return None

    return pisada_idx


# ============================================================================
# OPCIÓN 6: CALCULAR PARÁMETROS PARA PACIENTE CAD
# ============================================================================


def opcion_parametros_cad():
    """
    OPCIÓN 6: Calcula parámetros biomecánicos completos para un paciente de CAD.
    
    Proceso interactivo:
        1. Usuario ingresa ID del paciente (ej: C01, C02, HV01)
        2. El sistema detecta automáticamente el tipo de paciente:
           - Pacientes clínicos (C01-C10): 500 fps (2 ms/frame)
           - Voluntarios Hallux Valgus (HV01-HV05): 200 Hz
        3. Busca todos los archivos .nii en CADDataset/{paciente_id}/
        4. Procesa cada archivo .nii:
           - Carga secuencia temporal completa
           - Endereza frames usando PCA
           - Calcula CoF y CoP para cada trial
        5. Genera dos CSVs:
           - Footsteps_detail.csv: Detalle de cada trial
           - Summary.csv: Resumen por lado (Left, Right, Overall)
        6. Opcionalmente genera GIF de un trial específico:
           - Muestra lista de archivos .nii disponibles
           - Usuario selecciona un trial
           - Genera GIF animado con trayectoria CoP
    
    Detección automática de frecuencia:
        - Si ID empieza con 'H': 200 Hz (Hallux Valgus)
        - Si ID empieza con 'C': 500 fps (Pacientes clínicos)
    
    Módulos utilizados:
        - Parameters_CAD.py: export_parameters_from_directory()
        - GIF_Generator.py: create_gif_cad() (opcional)
    
    Salida:
        - Directorio: Salida_CAD/{paciente_id}/
        - Archivos: Footsteps_detail.csv, Summary.csv
        - GIF (opcional): cop_trajectory_{nombre_trial}.gif
    
    Parsing automático de nombres:
        - Extrae lado (left/right) del nombre del archivo
        - Extrae número de trial del nombre
        - Ejemplo: "left_foot_trial_01.nii" → Side=left, Trial=1
    
    Métricas en CSV:
        - File name: Nombre del archivo .nii
        - Side: left/right (detectado automáticamente)
        - Trial number: Número de ensayo
        - Stance time (s): Duración calculada como frames/fps
        - Num. frames valid: Frames con presión > 0
        - Mean CoF por región (%): Distribución promedio de fuerza
        - Frames CoP por región (%): Porcentaje de tiempo del CoP en cada región
    
    Note:
        - Los frames se enderezan automáticamente (effective_invert=False para CAD)
        - Ignora archivos fantasma de macOS (._*)
        - El resumen agrega métricas por lado y overall
    """
    print("\n" + "=" * 60)
    print("CALCULAR PARÁMETROS - CAD")
    print("=" * 60)

    paciente_text = input(
        "\nIngresa el ID del paciente (ej. C01, C02, C03): "
    ).strip().upper()

    if not paciente_text:
        print("❌ ID de paciente inválido.")
        return

    cad_directory = Path("CADDataset") / paciente_text
    if not cad_directory.exists():
        print(f"❌ No se encontró el directorio: {cad_directory}")
        print("   Verifica que el ID del paciente sea correcto.")
        return

    nii_files = list(cad_directory.glob("*.nii"))
    if not nii_files:
        print(f"❌ No se encontraron archivos .nii en: {cad_directory}")
        return

    print(f"\n📊 Encontrados {len(nii_files)} archivos .nii para paciente {paciente_text}")
    print(f"📁 Directorio: {cad_directory}")

    # Determinar frecuencia de muestreo según el tipo de paciente
    # HV (Healthy Volunteer) usa 500 Hz, pacientes C usan 500 fps (2ms por frame)
    if paciente_text.startswith('H'):
        fps = 200.0
        tipo_paciente = "Voluntario Hallux Valgus (HV)"
        print(f"📊 Tipo: {tipo_paciente} → Frecuencia: {fps} Hz")
    else:
        fps = 500.0
        tipo_paciente = "Paciente Clínico (C)"
        print(f"📊 Tipo: {tipo_paciente} → Frecuencia: {fps} fps (2ms/frame)")

    output_dir = Path("Salida_CAD") / paciente_text
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        detail_path, summary_path = export_parameters_cad(
            directory=cad_directory,
            output_dir=output_dir,
            pattern="*.nii",
            fps=fps,
            ratios=DEFAULT_RATIOS,
        )

        print(f"\n✅ CSV generados para paciente {paciente_text}:")
        print(f"   📄 Detalle: {detail_path}")
        print(f"   📄 Resumen: {summary_path}")

        manejar_visualizaciones_cad(output_dir, cad_directory, fps, paciente_text)

    except Exception as e:
        print(f"\n❌ Error al generar parámetros: {e}")


def manejar_visualizaciones_cad(output_dir: Path, cad_directory: Path, fps: float, paciente_id: str):
    """
    Maneja la generación opcional de GIFs para CAD después de calcular parámetros.
    
    Proceso interactivo:
        1. Pregunta al usuario si desea generar un GIF
        2. Si acepta:
           a. Lista todos los archivos .nii disponibles (ignora archivos fantasma ._*)
           b. Muestra lista numerada de archivos
           c. Usuario selecciona un archivo por número
           d. Genera GIF de la trayectoria CoP de ese trial
    
    Args:
        output_dir: Directorio donde guardar el GIF
        cad_directory: Directorio con archivos .nii del paciente
        fps: Frecuencia de muestreo (500 fps para C, 200 Hz para HV)
        paciente_id: ID del paciente (para mostrar unidades correctas)
    
    Información mostrada:
        - Lista de archivos .nii con índice numérico
        - Si hay más de 15 archivos: muestra primeros 10 y últimos 5
        - Muestra unidades correctas (Hz para HV, fps para C)
    
    GIF generado:
        - Nombre: cop_trajectory_{nombre_trial}.gif
        - Contenido: Animación de presión + trayectoria CoP
        - Configuración: 20 fps, ratios DEFAULT_RATIOS, frecuencia correcta
    
    Note:
        - Filtra archivos fantasma de macOS (._*)
        - Usa el fps correcto según tipo de paciente
        - Requiere matplotlib y nibabel instalados
    """
    print("\n" + "-" * 60)
    generar_gif = input("¿Desea generar un GIF de la trayectoria CoF? (s/n): ").strip().lower()

    if generar_gif not in ['s', 'si', 'sí', 'yes', 'y']:
        return

    # Filtrar archivos .nii válidos (excluir archivos fantasma de macOS que empiezan con ._)
    nii_files = sorted([f for f in cad_directory.glob("*.nii") if not f.name.startswith("._")])

    print(f"\n📋 Archivos .nii disponibles ({len(nii_files)} en total):")
    print("-" * 60)
    if len(nii_files) <= 15:
        for idx, nii_file in enumerate(nii_files, 1):
            print(f"  [{idx:2d}] {nii_file.name}")
    else:
        for idx, nii_file in enumerate(nii_files[:10], 1):
            print(f"  [{idx:2d}] {nii_file.name}")
        print("  ...")
        print(f"  (Mostrando 10 de {len(nii_files)} archivos)")
        print("\n  Últimos archivos:")
        for idx, nii_file in enumerate(nii_files[-5:], len(nii_files) - 4):
            print(f"  [{idx:2d}] {nii_file.name}")

    print("-" * 60)
    archivo_text = input(f"\nIngresa el número de archivo (1-{len(nii_files)}): ").strip()
    if not archivo_text.isdigit():
        print("❌ Número inválido.")
        return

    archivo_idx = int(archivo_text)
    if not (1 <= archivo_idx <= len(nii_files)):
        print(f"❌ Número fuera de rango. Debe estar entre 1 y {len(nii_files)}.")
        return

    selected_file = nii_files[archivo_idx - 1]
    print(f"\n✓ Seleccionado: {selected_file.name}")

    try:
        from GIF_Generator import create_gif_cad

        gif_path = output_dir / f"cop_trajectory_{selected_file.stem}.gif"
        
        # Usar el fps correcto según el tipo de paciente
        print(f"\n🎬 Generando GIF con frecuencia: {fps} {'Hz' if paciente_id.startswith('H') else 'fps'}...")
        
        create_gif_cad(
            nii_file_path=selected_file,
            output_path=gif_path,
            ratios=DEFAULT_RATIOS,
            fps=fps,
            fps_gif=20,
        )
        print(f"\n✅ GIF generado exitosamente!")
        print(f"   🎬 Ubicación: {gif_path}")
        print(f"   📏 Tamaño: {gif_path.stat().st_size / 1024:.1f} KB")
    except ImportError:
        print("\n⚠️  No se pudo generar el GIF: matplotlib no está instalado.")
        print("   Instala con: pip install matplotlib")
    except Exception as e:
        print(f"\n❌ Error al generar GIF: {e}")


# ============================================================================
# UTILIDAD PARA CONTINUAR
# ============================================================================


def preguntar_continuar() -> bool:
    """
    Pregunta al usuario si desea realizar otra operación o salir del programa.
    
    Función de control de flujo que se llama después de completar cada opción
    del menú principal. Valida la entrada del usuario y solo acepta respuestas
    válidas (s/n y variantes).
    
    Returns:
        bool: True si el usuario quiere continuar, False si quiere salir
    
    Respuestas aceptadas:
        - Para continuar: 's', 'si', 'sí', 'yes', 'y'
        - Para salir: 'n', 'no'
        - Otras: Pide entrada nuevamente
    """
    while True:
        print("\n" + "-" * 60)
        respuesta = input("¿Desea realizar otra operación? (s/n): ").strip().lower()
        if respuesta in ['s', 'si', 'sí', 'yes', 'y']:
            return True
        if respuesta in ['n', 'no']:
            print("\n" + "=" * 60)
            print("✨ ¡Muchas gracias por utilizar este algoritmo! ✨")
            print("=" * 60)
            return False
        print("❌ Por favor, responde 's' para sí o 'n' para no.")


# ============================================================================
# MENÚ PRINCIPAL
# ============================================================================


def main():
    """
    Función principal - Menú interactivo de la aplicación.
    
    Punto de entrada del programa. Muestra un menú con 7 opciones y ejecuta
    la funcionalidad seleccionada por el usuario en un bucle continuo hasta
    que el usuario decida salir.
    
    Flujo de ejecución:
        1. Inicializa ventana oculta de tkinter (para diálogos de archivos)
        2. Muestra menú principal con opciones numeradas 1-7
        3. Lee y valida la elección del usuario
        4. Ejecuta la función correspondiente a la opción elegida
        5. Pregunta si desea continuar o salir
        6. Repite hasta que el usuario elija salir (opción 7 o responder 'n')
    
    Opciones disponibles:
        1. Extraer imágenes desde .mat (CASIA)
        2. Extraer imágenes peak desde .nii (CAD)
        3. Extraer imágenes peak de StepUp (.npz)
        4. Analizar una imagen individual
        5. Calcular parámetros para paciente StepUp
        6. Calcular parámetros para paciente CAD
        7. Salir del programa
    
    Manejo de errores:
        - Opciones inválidas muestran mensaje de error y repiten el menú
        - Errores en las funciones se capturan y reportan sin cerrar el programa
        - Ctrl+C permite salir en cualquier momento
    
    Note:
        - La ventana de tkinter se mantiene oculta (root.withdraw())
        - Solo se usa para mostrar diálogos de selección de archivos
        - El programa es completamente interactivo (no requiere argumentos CLI)
    """
    root = tk.Tk()
    root.withdraw()

    while True:
        print("\n" + "=" * 60)
        print("       MENÚ PRINCIPAL - PROCESAMIENTO DE IMÁGENES PLANTARES")
        print("=" * 60)
        print("  1. Extraer imágenes desde archivo .mat (CASIA)")
        print("  2. Extraer imágenes peak desde carpeta .nii (CAD)")
        print("  3. Extraer imágenes peak de StepUp (.npz)")
        print("  4. Analizar una imagen individual")
        print("  5. Calcular parámetros para paciente StepUp")
        print("  6. Calcular parámetros para paciente CAD")
        print("  7. Salir del programa")
        print("=" * 60)

        choice = input("\nIngresa tu elección (1-7): ").strip()

        if choice == '1':
            opcion_extraer_mat()
            if not preguntar_continuar():
                break

        elif choice == '2':
            opcion_extraer_nii()
            if not preguntar_continuar():
                break

        elif choice == '3':
            opcion_extraer_stepup()
            if not preguntar_continuar():
                break

        elif choice == '4':
            opcion_analizar_imagen()
            if not preguntar_continuar():
                break

        elif choice == '5':
            opcion_parametros_stepup()
            if not preguntar_continuar():
                break

        elif choice == '6':
            opcion_parametros_cad()
            if not preguntar_continuar():
                break

        elif choice == '7':
            print("\n" + "=" * 60)
            print("✨ ¡Muchas gracias por utilizar este algoritmo! ✨")
            print("=" * 60)
            break

        else:
            print("\n" + "!" * 60)
            print("❌ ERROR: Opción no válida. Por favor, ingresa un número del 1 al 7.")
            print("!" * 60)


if __name__ == "__main__":
    main()

