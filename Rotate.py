#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de enderezamiento y segmentación de huellas plantares.

DESCRIPCIÓN:
    Este módulo contiene todas las funciones necesarias para enderezar imágenes
    y secuencias de frames de presiones plantares. Utiliza técnicas de visión
    por computadora (OpenCV) y análisis de componentes principales (PCA) para:
    1. Detectar y segmentar el pie del fondo
    2. Calcular el ángulo de inclinación
    3. Enderezar la imagen/frames para que el eje longitudinal quede vertical
    4. Dividir el pie en regiones (retropié, mediopié, antepié, dedos)

CONFIGURACIONES POR DATASET:
    El módulo soporta 3 configuraciones predefinidas según el dataset:
    
    CAD (Footscan):
        - Kernel cierre: 25×25 (rellena huecos grandes)
        - Kernel apertura: 0 (sin filtro de ruido, imágenes limpias)
        - Grosor líneas: 5px (box), 2px (divisiones)
        - Uso: Imágenes de alta calidad del sistema Footscan
    
    CASIA-D:
        - Kernel cierre: 6×6 (rellena huecos pequeños)
        - Kernel apertura: 5×5 (elimina ruido considerable)
        - Grosor líneas: 1px (box y divisiones)
        - Uso: Imágenes con más ruido, requieren filtrado
    
    STEPUP:
        - Misma configuración que CAD (25×25, 0, 5px, 2px)
        - Uso: Frames de secuencias temporales del StepUp-P150

FUNCIONES PRINCIPALES:

    Para imágenes individuales (Opción 4 de main.py):
        - enderezar_por_contorno(): Endereza una imagen estática desde archivo
        - procesar_huella_desde_ruta(): Procesamiento completo con visualización
        - enmarcar_y_dividir_huella(): Dibuja líneas divisorias de regiones

    Para secuencias de frames (Opciones 5 y 6 de main.py):
        - straighten_pressure_frames(): Endereza un stack completo de frames
        - Retorna StraightenedStack con frames enderezados y metadatos

TÉCNICAS UTILIZADAS:

    1. Segmentación:
        - Detección de color de fondo
        - Máscara binaria con umbral adaptativo
        - Morfología: OPEN (elimina ruido) + CLOSE (rellena huecos)
    
    2. Enderezamiento:
        - PCA (Principal Component Analysis) para calcular eje principal
        - Rotación usando cv2.warpAffine para alinear el eje vertical
        - Recorte espacial al área de interés (spatial crop)
    
    3. División en regiones:
        - Detección automática de orientación (talón arriba/abajo)
        - División en 4 regiones según ratios: (0-30%, 30-55%, 55-85%, 85-100%)
        - Líneas verdes horizontales marcan las divisiones

CLASES DE DATOS:
    - StraightenedStack: Contiene frames enderezados + metadatos de transformación

USO EN EL PROYECTO:
    - main.py Opción 4: Endereza imagen individual para análisis estático
    - Parameters_StepUp.py: Endereza secuencias de frames de StepUp
    - Parameters_CAD.py: Endereza secuencias de frames de CAD
    - GIF_Generator.py: Endereza frames antes de generar GIFs

FLUJO DE PROCESAMIENTO:
    Imagen/Frames → Segmentación → PCA → Rotación → Recorte → División en regiones
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


_DB_CONFIG: Dict[str, Dict[str, int]] = {
    "CAD": {"close": 25, "open": 0, "box": 5, "line": 2},
    "CASIA": {"close": 6, "open": 5, "box": 1, "line": 1},
    "STEPUP": {"close": 25, "open": 0, "box": 5, "line": 2},
}


def _get_db_config(nombre_db: str) -> Dict[str, int]:
    key = (nombre_db or "CAD").strip().upper()
    return _DB_CONFIG.get(key, _DB_CONFIG["CAD"])


@dataclass
class StraightenedStack:
    """
    Resultado del enderezamiento de una secuencia de frames.
    
    Contiene los frames enderezados y todos los metadatos necesarios para
    análisis posterior (orientación, ángulo de rotación, etc.).
    
    Attributes:
        frames: Lista de arrays numpy con los frames enderezados
        mask: Máscara binaria de la proyección máxima (después de morfología)
        angle: Ángulo de rotación aplicado en grados
        warp_matrix: Matriz de transformación 2×3 usada para la rotación
        output_size: Tamaño (ancho, alto) de los frames enderezados
        is_horizontal: True si el eje longitudinal del pie es horizontal
        invert: True si el talón está hacia la derecha (horizontal) o arriba (vertical)
                Este flag indica si se necesita inversión lógica en las máscaras de región
    """
    frames: List[np.ndarray]
    mask: np.ndarray
    angle: float
    warp_matrix: np.ndarray
    output_size: Tuple[int, int]
    is_horizontal: bool
    invert: bool


def _projection_threshold(projection: np.ndarray, mask_threshold: Optional[float]) -> float:
    if mask_threshold is not None:
        return float(mask_threshold)
    positives = projection[projection > 0]
    if positives.size == 0:
        return 0.0
    return float(np.percentile(positives, 5))


def _build_binary_mask(projection: np.ndarray, threshold: float) -> np.ndarray:
    mask = (projection > threshold).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("La proyección debe ser una imagen 2D.")
    return mask * 255


def _contour_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float], float]]:
    points = cv2.findNonZero(mask)
    if points is None or len(points) == 0:
        raise RuntimeError("No se encontraron píxeles válidos para enderezar.")
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    contour = np.intp(box)
    return contour, rect


def _warp_parameters_from_rect(
    rect: Tuple[Tuple[float, float], Tuple[float, float], float],
    contour: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    width_rect, height_rect = rect[1]
    if width_rect <= 0 or height_rect <= 0:
        raise RuntimeError("El rectángulo mínimo no es válido para enderezar.")

    src_points = order_points(contour.astype("float32"))
    width, height = width_rect, height_rect
    if width > height:
        width, height = height, width

    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    dst_points = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    angle = rect[2]
    if rect[1][0] >= rect[1][1]:
        angle += 90
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180

    return matrix, (width, height), angle


def _spatial_crop_stack(
    footstep: np.ndarray, thresh: float = 10
) -> Tuple[List[np.ndarray], Tuple[int, int, int, int]]:
    """
    Recorta espacialmente un stack de frames al área no-cero.
    
    Returns:
        cropped_frames: Lista de frames recortados
        bounds: (h_start, h_end, w_start, w_end)
    """
    img_peak = footstep.max(0)
    
    arr_w = np.sum(img_peak, axis=0)
    arr_h = np.sum(img_peak, axis=1)
    
    h_indices = np.where(arr_h > thresh)[0]
    w_indices = np.where(arr_w > thresh)[0]
    
    if h_indices.size == 0 or w_indices.size == 0:
        # No hay área válida, devolver todo
        return [footstep[i] for i in range(footstep.shape[0])], (0, footstep.shape[1]-1, 0, footstep.shape[2]-1)
    
    h_start = h_indices[0]
    h_end = h_indices[-1]
    w_start = w_indices[0]
    w_end = w_indices[-1]
    
    cropped = [footstep[i, h_start:h_end+1, w_start:w_end+1] for i in range(footstep.shape[0])]
    
    return cropped, (h_start, h_end, w_start, w_end)


def _orientation_from_mask(mask: np.ndarray) -> Tuple[bool, bool]:
    h, w = mask.shape
    is_horizontal = w >= h
    if is_horizontal:
        width = max(1, int(round(w * 0.2)))
        left = int(mask[:, :width].sum())
        right = int(mask[:, w - width :].sum())
        invert = right > left
    else:
        height = max(1, int(round(h * 0.2)))
        top = int(mask[:height, :].sum())
        bottom = int(mask[h - height :, :].sum())
        invert = top > bottom
    return is_horizontal, invert


def straighten_pressure_frames(
    frames: Sequence[np.ndarray],
    *,
    nombre_db: str = "CAD",
    mask_threshold: Optional[float] = None,
) -> StraightenedStack:
    """
    Endereza una secuencia completa de frames de presión plantar.
    
    Esta función es el corazón del enderezamiento de secuencias temporales.
    Utiliza la proyección máxima (peak) de todos los frames para calcular el
    ángulo de rotación óptimo mediante PCA, y luego aplica esa rotación a cada
    frame individual.
    
    Args:
        frames: Secuencia de arrays 2D numpy con frames de presión
        nombre_db: Nombre del dataset ("CAD", "CASIA", o "STEPUP") para usar
                  configuración específica de filtros morfológicos
        mask_threshold: Umbral manual para la máscara binaria. Si es None,
                       se calcula automáticamente como percentil 5 de valores > 0
    
    Returns:
        StraightenedStack: Objeto con frames enderezados y metadatos
    
    Proceso detallado:
        1. Proyección máxima: Calcula max(frames) para obtener huella completa
        2. Segmentación: Crea máscara binaria con umbral adaptativo
        3. Morfología: Aplica OPEN (elimina ruido) + CLOSE (rellena huecos)
        4. PCA: Calcula el eje principal de la huella
        5. Rotación: Aplica cv2.warpAffine a cada frame con padding temporal
        6. Recorte: Elimina áreas vacías (spatial crop)
        7. Orientación: Detecta si talón está arriba/abajo para máscaras de región
    
    Diferencias con enderezar_por_contorno():
        - enderezar_por_contorno(): Para imágenes individuales (PNG/JPG)
        - straighten_pressure_frames(): Para secuencias de frames (arrays numpy)
    
    Note:
        - Usa PCA en lugar de cv2.minAreaRect para mejor robustez
        - Mantiene el dtype original de los frames
        - El spatial crop reduce memoria y mejora eficiencia
        - La rotación es la misma para todos los frames (calculada del peak)
    """
    if not frames:
        raise ValueError("Se requiere al menos un frame para enderezar.")

    cfg = _get_db_config(nombre_db)
    base_dtype = np.asarray(frames[0]).dtype

    stack = [np.asarray(frame, dtype=np.float32) for frame in frames]
    stack_array = np.stack(stack, axis=0)
    
    # Proyección máxima para obtener la huella completa
    projection = np.max(stack_array, axis=0)
    threshold = _projection_threshold(projection, mask_threshold)
    mask = _build_binary_mask(projection, threshold)

    # Aplicar morfología
    if cfg["open"] > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg["open"], cfg["open"])
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    if cfg["close"] > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg["close"], cfg["close"])
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Calcular ángulo usando PCA (similar al código de referencia)
    c, r = np.where(projection > threshold)
    if c.shape[0] < 2:
        # No hay suficientes puntos, devolver frames sin rotar
        return StraightenedStack(
            frames=stack,
            mask=mask,
            angle=0.0,
            warp_matrix=np.eye(2, 3, dtype=np.float32),
            output_size=(projection.shape[1], projection.shape[0]),
            is_horizontal=projection.shape[1] >= projection.shape[0],
            invert=False,
        )
    
    # PCA para encontrar el eje principal
    X = np.array([c, r]).T
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered.T)
    eigval, eigvec = np.linalg.eig(cov)
    
    # Vector del primer componente principal
    v = eigvec[:, np.argmax(eigval)]
    angle_rad = np.arctan2(v[1], v[0])
    angle_deg = np.degrees(angle_rad)
    
    # Normalizar ángulo
    if abs(angle_deg) > 90:
        angle_deg = angle_deg - 180 * np.sign(angle_deg)
    
    rotation_angle = -angle_deg  # Ángulo de corrección
    
    # Calcular tamaño del canvas rotado
    h, w = projection.shape
    max_sz = int(np.ceil(np.sqrt(h**2 + w**2)))
    
    # Rotar cada frame
    rotated_frames: List[np.ndarray] = []
    for frame in stack_array:
        # Pad del frame al tamaño máximo (centrado)
        pad_top = (max_sz - frame.shape[0]) // 2
        pad_left = (max_sz - frame.shape[1]) // 2
        frame_padded = np.zeros((max_sz, max_sz), dtype=np.float32)
        frame_padded[pad_top:pad_top+frame.shape[0], pad_left:pad_left+frame.shape[1]] = frame
        
        # Matriz de rotación alrededor del centro
        M = cv2.getRotationMatrix2D((max_sz // 2, max_sz // 2), rotation_angle, 1.0)
        
        # Aplicar rotación
        frame_rotated = cv2.warpAffine(
            frame_padded, 
            M, 
            (max_sz, max_sz), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
        
        np.clip(frame_rotated, 0.0, None, out=frame_rotated)
        rotated_frames.append(frame_rotated)
    
    # Rotar la máscara también
    mask_padded = np.zeros((max_sz, max_sz), dtype=np.uint8)
    pad_top = (max_sz - mask.shape[0]) // 2
    pad_left = (max_sz - mask.shape[1]) // 2
    mask_padded[pad_top:pad_top+mask.shape[0], pad_left:pad_left+mask.shape[1]] = mask
    
    M = cv2.getRotationMatrix2D((max_sz // 2, max_sz // 2), rotation_angle, 1.0)
    rotated_mask = cv2.warpAffine(
        mask_padded,
        M,
        (max_sz, max_sz),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Recortar al área no-cero (spatial crop)
    rotated_stack_array = np.stack(rotated_frames, axis=0)
    cropped_stack, crop_bounds = _spatial_crop_stack(rotated_stack_array, thresh=threshold)
    
    # Recortar la máscara con los mismos límites
    h_start, h_end, w_start, w_end = crop_bounds
    cropped_mask = rotated_mask[h_start:h_end+1, w_start:w_end+1]
    
    # Convertir de vuelta al dtype original
    final_frames = [frame.astype(base_dtype, copy=False) for frame in cropped_stack]
    
    # Determinar orientación
    is_horizontal, invert = _orientation_from_mask(cropped_mask)

    return StraightenedStack(
        frames=final_frames,
        mask=cropped_mask,
        angle=rotation_angle,
        warp_matrix=M,
        output_size=(cropped_mask.shape[1], cropped_mask.shape[0]),
        is_horizontal=is_horizontal,
        invert=invert,
    )


def detectar_color_fondo(img):
    return img[5, 5].tolist()


def crear_mascara(img, color_fondo, thr=30):
    diff = np.abs(img.astype(int) - np.array(color_fondo))
    diff = np.sum(diff, axis=2)
    mascara = (diff > thr).astype(np.uint8)
    return mascara


def obtener_angulo_orientacion_ROBUSTO(contorno):
    pts = contorno.reshape(-1, 2).astype(np.float32)
    pts = cv2.convexHull(pts)
    media = np.mean(pts, axis=0)
    pts_centered = pts - media
    cov = np.cov(pts_centered.T)
    eigval, eigvec = np.linalg.eig(cov)
    idx = np.argmax(eigval)
    v = eigvec[:, idx]
    ang = np.degrees(np.arctan2(v[1], v[0]))
    if ang < -90: ang += 180
    if ang > 90: ang -= 180
    return ang


def obtener_angulo_orientacion(mascara):
    ys, xs = np.where(mascara == 1)
    puntos = np.column_stack([xs, ys]).astype(np.float32)
    media = np.mean(puntos, axis=0)
    cov = np.cov((puntos - media).T)
    eigval, eigvec = np.linalg.eig(cov)
    v = eigvec[:, np.argmax(eigval)]
    ang = np.degrees(np.arctan2(v[1], v[0]))
    return ang


def rotar_img_y_mascara(img, mascara, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1.0)
    img_r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    mask_r = cv2.warpAffine(mascara, M, (w, h), flags=cv2.INTER_NEAREST)
    return img_r, mask_r


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extraer_pie_recto(imagen, contorno, color_de_relleno):
    rect = cv2.minAreaRect(contorno)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    src_points = order_points(box.astype("float32"))
    (ancho, alto) = rect[1]
    if ancho > alto:
        ancho, alto = alto, ancho
    dst_points = np.array([
        [0, 0],
        [ancho - 1, 0],
        [ancho - 1, alto - 1],
        [0, alto - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    if len(imagen.shape) == 2:
        relleno_val = int(color_de_relleno[0]) if isinstance(color_de_relleno, (tuple, list, np.ndarray)) else int(color_de_relleno)
    else:
        relleno_val = tuple(map(int, color_de_relleno)) if isinstance(color_de_relleno, (tuple, list, np.ndarray)) else (int(color_de_relleno), int(color_de_relleno), int(color_de_relleno))
    img_warped = cv2.warpPerspective(imagen, M, (int(ancho), int(alto)),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=relleno_val)
    return img_warped


def enmarcar_y_dividir_huella(imagen_recta, mascara_recta, grosor_box_final, grosor_linea):
    h, w = imagen_recta.shape[:2]
    if h == 0 or w == 0:
        return imagen_recta
    img_dividida = imagen_recta.copy()
    zona_superior = mascara_recta[0 : int(h * 0.20)]
    zona_inferior = mascara_recta[int(h * 0.80) : h]
    masa_superior = cv2.countNonZero(zona_superior)
    masa_inferior = cv2.countNonZero(zona_inferior)
    talon_esta_abajo = masa_inferior < masa_superior
    pcts = [0.30, 0.55, 0.85]
    if talon_esta_abajo:
        print("Orientación detectada: Talón abajo (extremo angosto), Dedos arriba (extremo ancho).")
        y_cortes = [int(h * (1.0 - p)) for p in pcts]
    else:
        print("Orientación detectada: Talón arriba (extremo angosto), Dedos abajo (extremo ancho).")
        y_cortes = [int(h * p) for p in pcts]
    color_linea = (0, 255, 0)
    for y_corte in y_cortes:
        cv2.line(img_dividida, (0, y_corte), (w, y_corte), color_linea, grosor_linea)
    color_box = (0, 0, 255)
    cv2.rectangle(img_dividida, (0, 0), (w - 1, h - 1), color_box, grosor_box_final)
    return img_dividida


def dibujar_box_rotado(imagen, contorno, grosor_box_rotado):
    imagen_con_box = imagen.copy()
    rect = cv2.minAreaRect(contorno)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(imagen_con_box, [box], 0, (0, 0, 255), grosor_box_rotado)
    return imagen_con_box


def procesar_huella_desde_ruta(ruta_archivo, tamano_kernel_cierre, tamano_kernel_apertura, 
                                grosor_del_box, grosor_linea, nombre_db='CAD', mostrar_plots=False):
    """
    Función controladora principal.
    Si nombre_db == 'CASIA', muestra explícitamente la máscara OPEN y la máscara CLOSE.
    """
    if not os.path.isfile(ruta_archivo):
        print(f"¡Error! El archivo no existe: {ruta_archivo}")
        return None, None
    
    # --- 1. IMAGEN ORIGINAL ---
    imagen_color_original = cv2.imread(ruta_archivo)
    if imagen_color_original is None:
        print(f"Error al cargar: {ruta_archivo}")
        return None, None
    
    # --- 2. MÁSCARA BINARIA (RAW) y FILTRADO (OPEN) ---
    color_fondo_bgr = imagen_color_original[0, 0]
    tolerancia = 10
    color_fondo_np = np.array(color_fondo_bgr, dtype=np.int16)
    limite_inferior = np.clip(color_fondo_np - tolerancia, 0, 255).astype(np.uint8)
    limite_superior = np.clip(color_fondo_np + tolerancia, 0, 255).astype(np.uint8)
    mascara_fondo = cv2.inRange(imagen_color_original, limite_inferior, limite_superior)
    mascara_binaria_raw = cv2.bitwise_not(mascara_fondo)
    
    # Aplicar OPEN (Apertura) -> ESTA ES LA MÁSCARA 'OPEN'
    if tamano_kernel_apertura > 0:
        print(f"Aplicando filtro 'Apertura' (OPEN) {tamano_kernel_apertura}x{tamano_kernel_apertura} para eliminar ruido.")
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tamano_kernel_apertura, tamano_kernel_apertura))
        mascara_binaria_open = cv2.morphologyEx(mascara_binaria_raw, cv2.MORPH_OPEN, open_kernel)
    else:
        mascara_binaria_open = mascara_binaria_raw
    
    area_pixeles = cv2.countNonZero(mascara_binaria_open)
    img_area_segmentada = cv2.bitwise_and(imagen_color_original,
                                          imagen_color_original,
                                          mask=mascara_binaria_open)
    
    # --- 3. MÁSCARA CERRADA (CLOSE) ---
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (tamano_kernel_cierre, tamano_kernel_cierre))
    mascara_cerrada_close = cv2.morphologyEx(mascara_binaria_open, cv2.MORPH_CLOSE, kernel_cierre, iterations=2)
    
    # IMPORTANTE: Trabajar con TODA la máscara, no solo un contorno individual
    # Obtener TODOS los puntos no-cero de la máscara
    puntos_mascara = cv2.findNonZero(mascara_cerrada_close)
    
    if puntos_mascara is None or len(puntos_mascara) == 0:
        print("¡Error! No se encontraron píxeles en la máscara.")
        return None, None
    
    # Calcular el rectángulo mínimo que engloba TODOS los puntos de la máscara
    # Esto incluye todos los pies/objetos detectados
    rect_completo = cv2.minAreaRect(puntos_mascara)
    
    # Crear un "contorno" virtual que representa toda la región de la máscara
    box_completo = cv2.boxPoints(rect_completo)
    contorno_completo = np.intp(box_completo)
    
    # Calcular orientación PCA de TODA la región detectada
    pts = puntos_mascara.reshape(-1, 2).astype(np.float32)
    # Submuestrear si hay demasiados puntos (para eficiencia)
    if len(pts) > 5000:
        indices = np.random.choice(len(pts), 5000, replace=False)
        pts_sample = pts[indices]
    else:
        pts_sample = pts
    
    media = np.mean(pts_sample, axis=0)
    pts_centered = pts_sample - media
    cov = np.cov(pts_centered.T)
    eigval, eigvec = np.linalg.eig(cov)
    v = eigvec[:, np.argmax(eigval)]
    angulo_pca = np.degrees(np.arctan2(v[1], v[0]))
    if angulo_pca < -90: angulo_pca += 180
    if angulo_pca > 90: angulo_pca -= 180
    
    print(f"Procesando TODA la máscara detectada ({len(puntos_mascara)} píxeles)")
    
    # --- 4. IMAGEN CON BOX ROTADO ---
    img_con_box_rotado = imagen_color_original.copy()
    cv2.drawContours(img_con_box_rotado, [contorno_completo], 0, (0, 0, 255), grosor_del_box)
    
    # --- 5. IMAGEN FINAL ---
    # Extraer TODA la región de la máscara (no solo un contorno)
    img_pie_recortado_y_recto = extraer_pie_recto(imagen_color_original, contorno_completo, color_fondo_bgr)
    mascara_recta = extraer_pie_recto(mascara_cerrada_close, contorno_completo, 0)
    img_final_dividida = enmarcar_y_dividir_huella(img_pie_recortado_y_recto, mascara_recta, grosor_del_box, grosor_linea)
    
    largo_pixeles = img_final_dividida.shape[0]
    
    # Cálculo del Ángulo
    (cx, cy), (w_rect, h_rect), ang = rect_completo
    if w_rect < h_rect:
        angulo_inclinacion = ang
    else:
        angulo_inclinacion = ang + 90
    if angulo_inclinacion > 90: angulo_inclinacion -= 180
    if angulo_inclinacion < -90: angulo_inclinacion += 180
    
    print(f"Ángulo de inclinación (corregido): {angulo_inclinacion:.2f}°")
    print("\n--- Métricas Calculadas ---")
    print(f"Área de contacto del pie: {area_pixeles} píxeles cuadrados")
    print(f"Largo del pie (altura del box): {largo_pixeles} píxeles")
    
    # --- MOSTRAR RESULTADOS CON LÓGICA CONDICIONAL ---
    if mostrar_plots:
        print(f"\nMostrando bitácora de procesamiento para: {nombre_db}")
        fig, axes = plt.subplots(1, 6, figsize=(36, 7))
        
        # Visualización para CASIA (Muestra OPEN y CLOSE explícitamente)
        if nombre_db == 'CASIA':
            # 1. Original
            axes[0].imshow(cv2.cvtColor(imagen_color_original, cv2.COLOR_BGR2RGB))
            axes[0].set_title("1. Imagen Original")
            # 2. Máscara OPEN (Para ver limpieza de ruido)
            axes[1].imshow(mascara_binaria_open, cmap='gray')
            axes[1].set_title(f"2. Máscara OPEN (Ruido Eliminado)")
            # 3. Máscara CLOSE (Para ver relleno de huecos)
            axes[2].imshow(mascara_cerrada_close, cmap='gray')
            axes[2].set_title(f"3. Máscara CLOSE (Huecos Rellenos)")
            # 4. Píxeles contados (Visualización del área)
            axes[3].imshow(cv2.cvtColor(img_area_segmentada, cv2.COLOR_BGR2RGB))
            axes[3].set_title("4. Área Segmentada (Color)")
            # 5. Box Rotado
            axes[4].imshow(cv2.cvtColor(img_con_box_rotado, cv2.COLOR_BGR2RGB))
            axes[4].set_title("5. Box Rotado")
            # 6. Final
            axes[5].imshow(cv2.cvtColor(img_final_dividida, cv2.COLOR_BGR2RGB))
            axes[5].set_title("6. Final Dividida")
        
        # Visualización para CAD (Estándar original)
        else:
            axes[0].imshow(cv2.cvtColor(imagen_color_original, cv2.COLOR_BGR2RGB))
            axes[0].set_title("1. Imagen Original")
            axes[1].imshow(mascara_binaria_open, cmap='gray')
            axes[1].set_title("2. Máscara de Área (Limpia)")
            axes[2].imshow(cv2.cvtColor(img_area_segmentada, cv2.COLOR_BGR2RGB))
            axes[2].set_title("3. Píxeles de Área (Contados)")
            axes[3].imshow(mascara_cerrada_close, cmap='gray')
            axes[3].set_title(f"4. Máscara 'Cerrada' (CLOSE)")
            axes[4].imshow(cv2.cvtColor(img_con_box_rotado, cv2.COLOR_BGR2RGB))
            axes[4].set_title(f"5. Box Rotado")
            axes[5].imshow(cv2.cvtColor(img_final_dividida, cv2.COLOR_BGR2RGB))
            axes[5].set_title(f"6. Final Dividida")
        
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    return img_final_dividida, angulo_inclinacion


def enderezar_por_contorno(ruta_imagen, nombre_db='CAD', mostrar_plots=False):
    """
    Endereza una imagen estática de huella plantar desde archivo.
    
    Esta es la función principal para procesar imágenes individuales (PNG, JPG, etc.).
    Es llamada desde main.py Opción 4 (Analizar una imagen individual).
    
    Args:
        ruta_imagen: Ruta al archivo de imagen (PNG, JPG, BMP, etc.)
        nombre_db: Nombre del dataset para usar configuración específica:
                  - 'CAD': Footscan (kernel 25×25, sin apertura)
                  - 'CASIA': CASIA-D (kernel 6×6, apertura 5×5)
                  - 'STEPUP': StepUp-P150 (misma config que CAD)
        mostrar_plots: Si True, muestra 6 imágenes del proceso de enderezamiento:
                      1. Imagen original
                      2. Máscara de área (limpia)
                      3. Píxeles de área (contados)
                      4. Máscara cerrada (CLOSE)
                      5. Box rotado
                      6. Final dividida (con líneas de regiones)
    
    Returns:
        Tuple[np.ndarray, float]:
            - imagen_enderezada: Imagen enderezada con líneas divisorias verdes
            - angulo: Ángulo de corrección aplicado en grados
    
    Proceso:
        1. Carga configuración según dataset
        2. Lee imagen desde archivo
        3. Segmenta el pie del fondo usando color de referencia
        4. Aplica morfología (OPEN + CLOSE)
        5. Calcula ángulo de inclinación
        6. Endereza usando transformación perspectiva
        7. Dibuja líneas divisorias de regiones (30%, 55%, 85%)
        8. Opcionalmente muestra visualización del proceso
    
    Note:
        - Las líneas verdes dividen el pie en 4 regiones anatómicas
        - El box rojo marca el contorno del pie enderezado
        - La visualización es útil para debugging y presentaciones
    """
    nombre_db_key = (nombre_db or "CAD").strip().upper()
    cfg = _get_db_config(nombre_db_key)

    if nombre_db_key == 'CASIA':
        print("Configuración CASIA seleccionada.")
    elif nombre_db_key == 'CAD':
        print("Configuración CAD seleccionada.")
    elif nombre_db_key == 'STEPUP':
        print("Configuración STEPUP seleccionada (usa parámetros CAD).")
    else:
        print(f"Base de datos '{nombre_db}' no reconocida. Usando configuración CAD por defecto.")
        nombre_db_key = 'CAD'
        cfg = _get_db_config(nombre_db_key)

    tamano_kernel_cierre = cfg["close"]
    tamano_kernel_apertura = cfg["open"]
    grosor_del_box = cfg["box"]
    grosor_linea = cfg["line"]
    
    print(f"Kernel Cierre (gap-fill): {tamano_kernel_cierre}x{tamano_kernel_cierre}")
    print(f"Kernel Apertura (noise-kill): {tamano_kernel_apertura}x{tamano_kernel_apertura}")
    print(f"Grosor: {grosor_del_box}px")
    
    return procesar_huella_desde_ruta(
        ruta_imagen,
        tamano_kernel_cierre,
        tamano_kernel_apertura,
        grosor_del_box,
        grosor_linea,
        nombre_db_key,
        mostrar_plots
    )



