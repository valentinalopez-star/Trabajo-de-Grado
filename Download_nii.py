#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracción y conversión de archivos .nii (NIfTI) a imágenes PNG y GIF.

DESCRIPCIÓN:
    Este módulo procesa archivos .nii (Neuroimaging Informatics Technology Initiative)
    que contienen secuencias temporales de presiones plantares del dataset CAD.
    Convierte los datos 3D (Altura × Ancho × Frames) en imágenes visualizables.

FORMATO NIfTI (.nii):
    - Formato estándar para datos volumétricos/temporales en neuroimagen
    - En este proyecto: contiene secuencias de presión plantar frame por frame
    - Estructura: Array 3D (H, W, F) donde F es el número de frames temporales
    - Header incluye metadatos como 'zooms' (resolución espacial en mm)

DATASET CAD (Footscan):
    - Frecuencia: 500 Hz (2 ms por frame)
    - Resolución espacial: 7.62 mm × 5.08 mm por píxel (píxeles rectangulares)
    - Cada archivo .nii representa un trial completo de una pisada

USO EN EL PROYECTO:
    - Opción 2 de main.py: "Extraer imágenes peak desde carpeta .nii (CAD)"
    - Procesa directorios completos de pacientes (ej: CADDataset/C01/)
    - Replica la estructura de carpetas en el directorio de salida

TIPOS DE EXPORTACIÓN:
    1. Peak PNG: Imagen de presión máxima (max de todos los frames)
    2. Frames individuales: Cada frame como PNG separado
    3. GIF animado: Secuencia completa como animación

FUNCIONES PRINCIPALES:
    - load_nifti_trial(): Carga un archivo .nii y reorganiza dimensiones
    - save_peak_png(): Guarda la imagen de presión máxima
    - save_all_frames(): Guarda frames individuales
    - save_gif(): Crea GIF animado de la secuencia
    - descargar_img(): Función principal que procesa directorios completos
    - is_valid_nii(): Filtra archivos .nii válidos (ignora archivos fantasma de macOS)

CARACTERÍSTICAS ESPECIALES:
    - Ignora archivos fantasma de macOS que empiezan con ._
    - Soporta .nii y .nii.gz (comprimidos)
    - Aplica rotación de 90° para visualización correcta
    - Usa información de 'zooms' del header para aspect ratio correcto
    - Replica estructura de carpetas del input en el output

EJEMPLO DE USO:
    Input:  CADDataset/C01/left_foot_trial_01.nii
    Output: imgs_C01/C01/left_foot_trial_01.png (peak)
            imgs_C01/C01/left_foot_trial_01.gif (animación)
"""

from pathlib import Path
from typing import Union, Tuple, List
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# -------------------------------
# Helpers de visualización/IO
# -------------------------------

def load_nifti_trial(nii_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple]:
    """
    Carga un archivo .nii y reorganiza las dimensiones para procesamiento temporal.
    
    Los archivos .nii del CAD dataset tienen formato (H, W, F) donde F es el número
    de frames. Esta función los reorganiza a (F, H, W) para facilitar el procesamiento
    frame por frame.
    
    Args:
        nii_path: Ruta al archivo .nii o .nii.gz
    
    Returns:
        Tuple[np.ndarray, Tuple]:
            - trial: Array 3D con formato (F, H, W) donde F=frames, H=altura, W=ancho
            - zooms: Tupla con la resolución espacial (dx, dy, dt) en mm
                    Para CAD: (7.62, 5.08, 2.0) = 7.62mm × 5.08mm por píxel, 2ms por frame
    
    Raises:
        ValueError: Si el archivo no tiene exactamente 3 dimensiones
    
    Note:
        - Convierte valores negativos a 0 (datos de presión no pueden ser negativos)
        - Convierte a float32 para eficiencia en procesamiento posterior
    """
    nii_path = Path(nii_path)
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    zooms = img.header.get_zooms()

    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"Esperaba 3D, obtuve shape {arr.shape} en {nii_path.name}")

    H, W, F = arr.shape
    trial = np.moveaxis(arr, -1, 0) if F >= max(H, W) else arr  # (F,H,W)
    trial = trial.astype(np.float32, copy=False)
    trial[trial < 0] = 0.0
    return trial, zooms


def save_peak_png(trial: np.ndarray,
                  out_png: Union[str, Path],
                  zooms: Tuple = None,
                  rotate: bool = True,
                  cmap: str = 'jet',
                  dpi: int = 150) -> Path:
    """
    Guarda la imagen de presión máxima (peak) de una secuencia temporal.
    
    Calcula el máximo de presión en cada píxel a lo largo de todos los frames
    y guarda el resultado como una imagen PNG.
    
    Args:
        trial: Array 3D (F, H, W) con la secuencia de frames de presión
        out_png: Ruta donde guardar la imagen PNG resultante
        zooms: Tupla con resolución espacial (dx, dy, dt) en mm. Si se proporciona,
              se usa para configurar el aspect ratio correcto de la visualización
        rotate: Si True, rota la imagen 90° para orientación correcta
        cmap: Colormap de matplotlib (por defecto 'jet' para presiones)
        dpi: Resolución de la imagen exportada (por defecto 150)
    
    Returns:
        Path: Ruta del archivo PNG guardado
    
    Note:
        - La imagen peak es útil para análisis estático de la pisada completa
        - Se usa figsize=(5, 6) y dpi=150 para balance entre calidad y tamaño
        - El colormap 'jet' es estándar para visualización de presiones (azul=bajo, rojo=alto)
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    peak = trial.max(axis=0)
    if rotate:
        peak = np.rot90(peak)

    fig, ax = plt.subplots(figsize=(5, 6), dpi=dpi)
    if zooms is not None and len(zooms) >= 2:
        dx, dy = float(zooms[0]), float(zooms[1])
        H, W = peak.shape
        extent = [0, W * dx, 0, H * dy]
        ax.imshow(peak, cmap=cmap, origin='lower', extent=extent, aspect=dx / dy)
    else:
        ax.imshow(peak, cmap=cmap, origin='lower')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return out_png


def save_all_frames(trial: np.ndarray,
                    out_dir: Union[str, Path],
                    prefix: str = "frame",
                    every: int = 1,
                    rotate: bool = True,
                    cmap: str = 'jet',
                    zooms: Tuple = None,
                    dpi: int = 120) -> List[Path]:
    """
    Guarda frames individuales de una secuencia temporal como imágenes PNG separadas.
    
    Útil para análisis frame-por-frame o creación de secuencias de imágenes.
    Permite submuestrear (guardar cada N frames) para reducir cantidad de archivos.
    
    Args:
        trial: Array 3D (F, H, W) con la secuencia de frames
        out_dir: Directorio donde guardar los frames
        prefix: Prefijo para los nombres de archivo (ej: "frame" → frame_0000.png)
        every: Guardar cada N frames (1=todos, 5=cada 5 frames, etc.)
        rotate: Si True, rota cada frame 90° para orientación correcta
        cmap: Colormap de matplotlib para visualización
        zooms: Resolución espacial (dx, dy, dt) en mm para aspect ratio
        dpi: Resolución de las imágenes exportadas
    
    Returns:
        List[Path]: Lista de rutas a todos los archivos PNG guardados
    
    Note:
        - Los nombres de archivo incluyen índice con 4 dígitos: frame_0000.png, frame_0001.png, etc.
        - Usar every > 1 es útil para reducir espacio en disco manteniendo información temporal
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    F = int(trial.shape[0])
    paths: List[Path] = []

    for i in range(0, F, max(1, int(every))):
        img = trial[i]
        if rotate:
            img = np.rot90(img)

        fig, ax = plt.subplots(figsize=(4, 5), dpi=dpi)
        if zooms is not None and len(zooms) >= 2:
            dx, dy = float(zooms[0]), float(zooms[1])
            H, W = img.shape
            extent = [0, W * dx, 0, H * dy]
            ax.imshow(img, cmap=cmap, origin='lower', extent=extent, aspect=dx / dy)
        else:
            ax.imshow(img, cmap=cmap, origin='lower')
        ax.axis('off')
        plt.tight_layout()

        out_png = out_dir / f"{prefix}_{i:04d}.png"
        fig.savefig(out_png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        paths.append(out_png)

    return paths


def save_gif(trial: np.ndarray,
             out_gif: Union[str, Path],
             rotate: bool = True,
             fps: int = 25) -> Path:
    """
    Crea un GIF animado de una secuencia temporal de presiones plantares.
    
    Genera una animación optimizada en 8 bits que muestra la evolución temporal
    de la presión plantar durante una pisada completa.
    
    Args:
        trial: Array 3D (F, H, W) con la secuencia de frames
        out_gif: Ruta donde guardar el archivo GIF resultante
        rotate: Si True, rota cada frame 90° para orientación correcta
        fps: Frames por segundo del GIF (velocidad de reproducción)
            Por defecto 25 fps para visualización fluida
    
    Returns:
        Path: Ruta del archivo GIF guardado
    
    Proceso:
        1. Normaliza todos los frames al rango 0-255 usando el máximo global
        2. Convierte cada frame a uint8 (8 bits)
        3. Rota frames si es necesario
        4. Crea GIF usando imageio con duración calculada desde fps
    
    Note:
        - Normalización global asegura que la escala de colores sea consistente
        - 8 bits reduce significativamente el tamaño del archivo
        - No incluye aspect ratio ni extent (versión simplificada para preview rápido)
        - Para GIFs con análisis de CoP, ver GIF_Generator.py
    """
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    F = int(trial.shape[0])
    vmax = float(trial.max()) if trial.max() > 0 else 1.0
    frames = []
    for i in range(F):
        img = trial[i]
        if rotate:
            img = np.rot90(img)
        frame8 = (np.clip(img / vmax, 0, 1) * 255).astype(np.uint8)
        frames.append(frame8)

    imageio.mimsave(out_gif, frames, duration=1.0 / float(fps))
    return out_gif


# -------------------------------
# Recorrer carpeta y exportar
# -------------------------------

def is_valid_nii(p: Path) -> bool:
    """
    Verifica si un archivo es un .nii válido y no es un archivo fantasma de macOS.
    
    macOS crea archivos ocultos que empiezan con ._ (ej: ._left_foot_trial_01.nii)
    para almacenar metadatos extendidos. Estos archivos deben ser ignorados.
    
    Args:
        p: Path al archivo a verificar
    
    Returns:
        bool: True si es un archivo .nii o .nii.gz válido (no fantasma)
    """
    if not p.is_file():
        return False
    if p.name.startswith("._"):
        return False
    if p.suffix == ".nii":
        return True
    # .nii.gz
    return (p.suffix == ".gz" and p.name.endswith(".nii.gz"))


def descargar_img(
    in_dir: Union[str, Path],
    out_dir: Union[str, Path],
    export_peak: bool = True,
    export_frames: bool = False,
    frames_every: int = 5,
    export_gif: bool = False,
    rotate: bool = True,
) -> None:
    """
    Procesa recursivamente un directorio con archivos .nii y exporta imágenes.
    
    Esta es la función principal del módulo. Busca todos los archivos .nii en
    el directorio de entrada (y subdirectorios), los procesa y guarda los resultados
    manteniendo la estructura de carpetas original.
    
    Args:
        in_dir: Directorio de entrada con archivos .nii (puede tener subdirectorios)
        out_dir: Directorio de salida donde se guardarán las imágenes
        export_peak: Si True, guarda imagen PNG de presión máxima para cada trial
        export_frames: Si True, guarda frames individuales como PNG separados
        frames_every: Intervalo de submuestreo para frames (1=todos, 5=cada 5, etc.)
        export_gif: Si True, crea GIF animado de cada trial
        rotate: Si True, rota imágenes 90° para orientación correcta
    
    Proceso:
        1. Busca recursivamente todos los archivos .nii válidos
        2. Para cada archivo:
           a. Carga el trial con load_nifti_trial()
           b. Replica la estructura de carpetas en out_dir
           c. Exporta según las opciones seleccionadas:
              - Peak PNG: Presión máxima de toda la secuencia
              - Frames: Imágenes individuales de cada frame
              - GIF: Animación de la secuencia completa
        3. Reporta progreso y maneja errores graciosamente
    
    Estructura de salida:
        in_dir/C01/left_foot_trial_01.nii
        → out_dir/C01/left_foot_trial_01.png (peak)
        → out_dir/C01/left_foot_trial_01_frames/ (frames individuales)
        → out_dir/C01/left_foot_trial_01.gif (animación)
    
    Note:
        - Ignora archivos fantasma de macOS (._*)
        - Soporta .nii y .nii.gz
        - Archivos con errores se omiten con warning (no detiene el proceso)
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    nii_files = [p for p in in_dir.rglob("*") if is_valid_nii(p)]
    if not nii_files:
        print(f"[INFO] No encontré .nii válidos en: {in_dir}")
        return

    print(f"[INFO] Encontrados {len(nii_files)} ensayos .nii")

    for nii in sorted(nii_files):
        rel = nii.relative_to(in_dir)           # p.ej. C01/left_foot_trial_01.nii
        base = rel.with_suffix("")              # sin .nii
        if base.name.endswith(".nii"):          # por si era .nii.gz
            base = Path(base.name[:-4])

        # carpetas de salida espejo
        out_subdir = out_dir / base.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        try:
            trial, zooms = load_nifti_trial(nii)

            if export_peak:
                out_png = out_dir / base.with_suffix(".png")
                save_peak_png(trial, out_png, zooms=zooms, rotate=rotate)
                print(f"[OK] Peak -> {out_png}")

            if export_frames:
                frames_dir = out_dir / base.parent / f"{base.name}_frames"
                save_all_frames(trial, frames_dir, prefix=base.name, every=frames_every, zooms=zooms, rotate=rotate)
                print(f"[OK] Frames -> {frames_dir}")

            if export_gif:
                out_gif = out_dir / base.with_suffix(".gif")
                save_gif(trial, out_gif, rotate=rotate, fps=25)
                print(f"[OK] GIF -> {out_gif}")

        except Exception as e:
            print(f"[WARN] Salteando {nii}: {e}")


