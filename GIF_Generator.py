#!/usr/bin/env python3
"""
Generador de GIFs animados para visualización de trayectorias del Centro de Presión (CoP).

DESCRIPCIÓN:
    Este módulo crea animaciones GIF que muestran la evolución temporal del Centro
    de Presión (CoP) durante una pisada completa. Cada frame del GIF muestra:
    - La distribución de presión en ese instante (mapa de calor)
    - La trayectoria acumulada del CoP hasta ese momento (línea verde)
    - Líneas divisorias de regiones del pie (líneas blancas punteadas)
    - Información temporal (frame actual y tiempo transcurrido)

DATASETS COMPATIBLES:
    1. StepUpDataset (.npz):
       - Datos de la pasarela StepUp-P150
       - Frecuencia: 100 Hz
       - Requiere: participant_id, shoe_type, walk_condition, pass_id, footstep_id
    
    2. CADDataset (.nii):
       - Datos del sistema Footscan
       - Frecuencia: 500 Hz (2 ms/frame) para pacientes C, 200 Hz para HV
       - Requiere: ruta al archivo .nii

USO EN EL PROYECTO:
    - main.py Opción 5: Genera GIF para pisada de StepUp
    - main.py Opción 6: Genera GIF para trial de CAD
    - Ambas opciones preguntan al usuario si desea generar el GIF después de calcular parámetros

FUNCIONES PRINCIPALES:
    - create_gif_stepup(): Crea GIF para una pisada específica de StepUp
    - create_gif_cad(): Crea GIF para un archivo .nii de CAD
    - _generate_gif(): Función interna común que genera el GIF
    - _calculate_region_lines(): Calcula posiciones de líneas divisorias
    - _cop(): Calcula CoP de un frame individual

CARACTERÍSTICAS DEL GIF:
    - Tamaño: 6×8 pulgadas (ajustable)
    - FPS: 20 fps por defecto (velocidad de reproducción)
    - Colormap: Jet con 0=negro (azul=baja presión, rojo=alta presión)
    - Trayectoria: Línea verde con puntos que muestra el recorrido del CoP
    - Líneas de región: Blancas punteadas en 30%, 55%, 85% del pie
    - Timestamp: Muestra frame actual y tiempo transcurrido

PROCESO DE GENERACIÓN:
    1. Carga datos (metadata + trial para StepUp, archivo .nii para CAD)
    2. Endereza frames usando straighten_pressure_frames()
    3. Detecta orientación y aplica inversión si es necesario
    4. Calcula trayectoria del CoP frame por frame
    5. Crea animación con matplotlib.animation
    6. Guarda como GIF usando PillowWriter

NOTA TÉCNICA:
    - Los frames se enderezan antes de calcular el CoP para consistencia
    - La trayectoria es acumulativa (muestra todo el recorrido hasta el frame actual)
    - origin='lower' en imshow para que y=0 esté abajo (convención estándar)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from Rotate import straighten_pressure_frames


# ============================================================================
# UTILIDADES COMUNES
# ============================================================================

def _cop(img: np.ndarray) -> Tuple[float, float]:
    """Calcula el Centro de Presión (CoP) de una imagen."""
    total = float(img.sum())
    if total <= 0:
        return float("nan"), float("nan")
    y_idx, x_idx = np.indices(img.shape)
    cop_x = float((img * x_idx).sum() / total)
    cop_y = float((img * y_idx).sum() / total)
    return cop_x, cop_y


def _create_colormap():
    """Crea colormap jet con 0=negro."""
    try:
        import matplotlib as mpl
        jet = mpl.colormaps['jet'](np.linspace(0, 1, 256))
        jet[0] = [0, 0, 0, 1]
        return mpl.colors.ListedColormap(jet)
    except Exception:
        return 'hot'  # Fallback


def _calculate_region_lines(
    shape: Tuple[int, int],
    is_horizontal: bool,
    effective_invert: bool,
    ratios: Sequence[float]
) -> List[float]:
    """
    Calcula las posiciones de las líneas de regiones.
    
    La lógica depende de cómo se normalizó en _region_masks:
    - Si effective_invert=True: norm = (axis - coord) / axis → líneas en (axis - 1) * (1 - ratio)
    - Si effective_invert=False: norm = coord / axis → líneas en (axis - 1) * ratio
    
    Para StepUp: effective_invert=False (después de aplicar inversión física)
    Para CAD: effective_invert=invert (sin inversión física)
    """
    H, W = shape
    
    if is_horizontal:
        axis_len = W - 1
        if effective_invert:
            line_pos = [axis_len * (1.0 - r) for r in ratios]
        else:
            line_pos = [axis_len * r for r in ratios]
    else:
        axis_len = H - 1
        if effective_invert:
            line_pos = [axis_len * (1.0 - r) for r in ratios]
        else:
            line_pos = [axis_len * r for r in ratios]
    
    return line_pos


# ============================================================================
# GENERADOR PARA STEPUP DATASET
# ============================================================================

def create_gif_stepup(
    participant_id: int,
    shoe_type: str,
    walk_condition: str,
    pass_id: int,
    footstep_id: int,
    output_path: Path | str,
    *,
    ratios: Sequence[float] = (0.30, 0.55, 0.85),
    fps_gif: int = 20,
) -> Path:
    """
    Genera un GIF de trayectoria CoP para el StepUpDataset.
    
    Args:
        participant_id: ID del participante
        shoe_type: Tipo de calzado (BF, ST, etc.)
        walk_condition: Condición de marcha (W1, W2, etc.)
        pass_id: Pass ID interno
        footstep_id: Footstep ID interno
        output_path: Ruta del GIF de salida
        ratios: Ratios para dividir el pie en regiones
        fps_gif: FPS del GIF
    
    Returns:
        Ruta del GIF generado
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter
    except ImportError as exc:
        raise ImportError(
            "Se requiere matplotlib. Instala con: pip install matplotlib"
        ) from exc
    
    try:
        from StepUpDataset.utils import load_metadata, load_trial
        from Parameters_StepUp import _extract_roi_stack, _get_fps
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar StepUpDataset.utils o Parameters_StepUp"
        ) from exc
    
    # Cargar datos
    metadata = load_metadata(participant_id, shoe_type, walk_condition)
    trial = load_trial(participant_id, shoe_type, walk_condition)
    
    # Buscar la pisada
    mask = (
        (metadata["Standing"] == 0)
        & (metadata["Incomplete"] == 0)
        & (metadata["PassID"] == pass_id)
        & (metadata["FootstepID"] == footstep_id)
    )
    steps = metadata[mask]
    
    if steps.empty:
        raise RuntimeError(
            f"No se encontró footstep PassID={pass_id}, FootstepID={footstep_id}"
        )
    
    step = steps.iloc[0]
    fps = _get_fps(step, 100.0)
    
    # Extraer y procesar footstep CON inversión física para StepUp
    roi_stack, start_f, end_f, is_horizontal, effective_invert = _extract_roi_stack(
        trial, step, apply_inversion=True  # Invertir físicamente
    )
    
    roi_ref = roi_stack[len(roi_stack) // 2]

    # Generar GIF
    return _generate_gif(
        roi_stack=roi_stack,
        is_horizontal=is_horizontal,
        effective_invert=effective_invert,  # Será False después de inversión
        ratios=ratios,
        fps=fps,
        fps_gif=fps_gif,
        output_path=output_path,
        title=f'Pass {pass_id} | Footstep {footstep_id} ({step["Side"]})'
    )


# ============================================================================
# GENERADOR PARA CAD DATASET
# ============================================================================

def create_gif_cad(
    nii_file_path: Path | str,
    output_path: Path | str,
    *,
    ratios: Sequence[float] = (0.30, 0.55, 0.85),
    fps: float = 500.0,
    fps_gif: int = 20,
) -> Path:
    """
    Genera un GIF de trayectoria CoP para el CADDataset.
    
    Args:
        nii_file_path: Ruta al archivo .nii
        output_path: Ruta del GIF de salida
        ratios: Ratios para dividir el pie en regiones
        fps: FPS del dataset (500 fps = 2ms por frame para CAD)
        fps_gif: FPS del GIF
    
    Returns:
        Ruta del GIF generado
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.animation import PillowWriter
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Se requiere matplotlib y nibabel. "
            "Instala con: pip install matplotlib nibabel"
        ) from exc
    
    try:
        from Parameters_CAD import _compute_orientation
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar Parameters_CAD"
        ) from exc
    
    # Cargar archivo .nii
    img = nib.load(str(nii_file_path))
    data = img.get_fdata()
    
    # Extraer y enderezar los frames
    roi_stack_raw = [data[:, :, i] for i in range(data.shape[2])]
    straightened = straighten_pressure_frames(roi_stack_raw, nombre_db="CAD")
    roi_stack = straightened.frames

    roi_ref = roi_stack[len(roi_stack) // 2]
    is_horizontal, invert = _compute_orientation(roi_ref)
    
    # Para CAD: NO invertir físicamente, NO usar inversión lógica
    effective_invert = False
    
    # Generar GIF
    file_name = Path(nii_file_path).stem
    return _generate_gif(
        roi_stack=roi_stack,
        is_horizontal=is_horizontal,
        effective_invert=effective_invert,  # False = máscaras directas
        ratios=ratios,
        fps=fps,
        fps_gif=fps_gif,
        output_path=output_path,
        title=f'CAD - {file_name}'
    )


# ============================================================================
# GENERADOR COMÚN
# ============================================================================

def _generate_gif(
    roi_stack: List[np.ndarray],
    is_horizontal: bool,
    effective_invert: bool,
    ratios: Sequence[float],
    fps: float,
    fps_gif: int,
    output_path: Path | str,
    title: str
) -> Path:
    """
    Genera el GIF con la trayectoria del CoP.
    
    Función interna que realiza la generación real del GIF.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter
    
    # Calcular trayectoria del CoP
    cop_trajectory = []
    for img in roi_stack:
        cop_x, cop_y = _cop(img)
        cop_trajectory.append((cop_x, cop_y))
    
    # Preparar visualización
    H, W = roi_stack[0].shape
    vmax = max(float(img.max()) for img in roi_stack)
    if vmax <= 0:
        vmax = 1.0
    
    # Crear colormap
    cmap = _create_colormap()
    
    # Calcular líneas de regiones
    line_pos = _calculate_region_lines(
        (H, W), is_horizontal, effective_invert, ratios
    )
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_title(f'Trayectoria CoP - {title}', fontsize=12)
    ax.set_xlabel('Ancho (píxeles)', fontsize=10)
    ax.set_ylabel('Largo (píxeles)', fontsize=10)
    
    # Dibujar líneas de regiones
    if is_horizontal:
        for pos in line_pos:
            ax.axvline(x=pos, color='white', linestyle='--', 
                      linewidth=1, alpha=0.5)
    else:
        for pos in line_pos:
            ax.axhline(y=pos, color='white', linestyle='--', 
                      linewidth=1, alpha=0.5)
    
    # Elementos animados
    im = ax.imshow(roi_stack[0], cmap=cmap, vmin=0, vmax=vmax, 
                   interpolation='nearest', origin='lower')
    line, = ax.plot([], [], 'lime', marker='o', markersize=3, 
                    linewidth=1.5, alpha=0.8)
    timestamp = ax.text(0.02, 0.98, '', color='white', fontsize=10,
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    def update_animation(frame_index):
        """Actualiza el frame de la animación."""
        # Actualizar imagen de presión
        im.set_data(roi_stack[frame_index])
        
        # Actualizar trayectoria del CoP (acumulativa)
        cop_x_hist = [cop_trajectory[i][0] for i in range(frame_index + 1) 
                      if np.isfinite(cop_trajectory[i][0])]
        cop_y_hist = [cop_trajectory[i][1] for i in range(frame_index + 1) 
                      if np.isfinite(cop_trajectory[i][1])]
        
        if cop_x_hist and cop_y_hist:
            line.set_data(cop_x_hist, cop_y_hist)
        
        # Actualizar timestamp
        time_s = frame_index / fps
        timestamp.set_text(f'Frame: {frame_index + 1}/{len(roi_stack)}\n'
                          f'Tiempo: {time_s:.3f}s')
        
        return im, line, timestamp
    
    # Crear animación
    ani = animation.FuncAnimation(
        fig, 
        update_animation, 
        frames=len(roi_stack),
        interval=1000/fps_gif,
        blit=True
    )
    
    # Guardar GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = PillowWriter(fps=fps_gif)
    ani.save(output_path, writer=writer)
    
    plt.close(fig)
    
    return output_path

