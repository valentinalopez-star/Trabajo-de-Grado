#!/usr/bin/env python3
"""
Utilidad para convertir dimensiones de píxeles a centímetros.

ESCALAS CONOCIDAS POR DATASET:

1. StepUp-P150:
   - Pasarela: 3.6 m × 1.2 m
   - Densidad: 4 sensores/cm² (2 px/cm lineal)
   - Escala: 1 px = 0.5 cm × 0.5 cm (píxeles CUADRADOS)
   - Frecuencia: 100 Hz
   - Dimensiones típicas: 720 × 240 px → 360 × 120 cm (3.6 × 1.2 m)

2. CAD Dataset (Footscan):
   - Placas: 0.5 m y 1.5 m
   - Escala según NIfTI zooms: (7.62, 5.08, 2.0) mm
   - 1 px = 0.762 cm (X) × 0.508 cm (Y) (píxeles RECTANGULARES)
   - Frecuencia: 500 Hz (2 ms/frame)
   - Dimensiones típicas: ~34 × 19 px → ~26 × 10 cm

3. CASIA-D:
   - Necesita verificación en CASgait.mat
   - Las imágenes se guardan sin redimensionar desde el .mat

IMPORTANTE:
- Las imágenes exportadas mantienen las dimensiones originales (NO se redimensionan)
- En main.py, la función normalize_image se llama con output_size=None, 
  lo que significa que SOLO ajusta brillo/contraste pero NO cambia el tamaño
- CAD tiene píxeles rectangulares (diferente escala en X e Y)
- Las escalas son correctas SOLO para datos crudos

⚠️ ADVERTENCIAS IMPORTANTES:
1. Las imágenes exportadas pueden incluir espacio en blanco alrededor del pie
2. Las imágenes de CAD se exportan con DPI=150 usando matplotlib, lo que redimensiona
   la imagen visualmente pero el tamaño físico real NO cambia
3. Para CALIBRACIÓN PRECISA se recomienda:
   - Usar datos crudos directamente (.nii, .npz)
   - O proporcionar una referencia conocida (ej: "mi pie mide 26 cm")
   - Usar: pixels_to_cm(..., reference_cm=26, reference_px=longitud_detectada_px)
4. La función calculate_foot_dimensions_from_pressure() hace una ESTIMACIÓN
   que puede variar según el proceso de exportación

NOTA SOBRE REDIMENSIONAMIENTO:
Si en algún momento se usa normalize_image(..., output_size=(new_w, new_h)),
las escalas cambiarían y necesitarías recalcularlas así:
  new_scale_x = old_scale_x * (old_width / new_width)
  new_scale_y = old_scale_y * (old_height / new_height)

Actualmente, como output_size=None, las escalas originales se mantienen.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np


def pixels_to_cm(
    width_px: float,
    height_px: float,
    scale_factor: Optional[float] = None,
    dpi: Optional[float] = None,
    reference_cm: Optional[float] = None,
    reference_px: Optional[float] = None
) -> Tuple[float, float]:
    """
    Convierte dimensiones de píxeles a centímetros.
    
    Métodos de conversión (elegir UNO):
    
    1. Usando scale_factor directo (píxeles por cm):
       scale_factor = píxeles / cm
       Ejemplo: si 100 píxeles = 10 cm → scale_factor = 10
    
    2. Usando DPI (dots per inch):
       dpi = píxeles por pulgada
       Ejemplo: 300 DPI significa 300 píxeles por pulgada
       1 pulgada = 2.54 cm
    
    3. Usando referencia conocida:
       reference_cm = tamaño real conocido en cm
       reference_px = tamaño en píxeles de ese objeto
       Ejemplo: si el pie mide 25 cm y ocupa 250 píxeles → reference_cm=25, reference_px=250
    
    Args:
        width_px: Ancho en píxeles
        height_px: Alto en píxeles
        scale_factor: Píxeles por centímetro (método 1)
        dpi: Píxeles por pulgada (método 2)
        reference_cm: Tamaño real de referencia en cm (método 3)
        reference_px: Tamaño en píxeles del objeto de referencia (método 3)
    
    Returns:
        Tuple[float, float]: (ancho en cm, alto en cm)
    
    Examples:
        >>> # Método 1: Scale factor directo
        >>> pixels_to_cm(100, 200, scale_factor=10)
        (10.0, 20.0)
        
        >>> # Método 2: Usando DPI
        >>> pixels_to_cm(300, 600, dpi=300)  # 300 DPI = ~118.11 px/cm
        (2.54, 5.08)
        
        >>> # Método 3: Usando referencia conocida
        >>> pixels_to_cm(250, 100, reference_cm=25, reference_px=250)
        (25.0, 10.0)
    """
    # Determinar el factor de conversión
    if scale_factor is not None:
        # Método 1: Scale factor directo
        px_per_cm = scale_factor
    elif dpi is not None:
        # Método 2: DPI (1 pulgada = 2.54 cm)
        px_per_cm = dpi / 2.54
    elif reference_cm is not None and reference_px is not None:
        # Método 3: Referencia conocida
        px_per_cm = reference_px / reference_cm
    else:
        raise ValueError(
            "Debes proporcionar uno de: scale_factor, dpi, o (reference_cm + reference_px)"
        )
    
    # Convertir
    width_cm = width_px / px_per_cm
    height_cm = height_px / px_per_cm
    
    return width_cm, height_cm


def get_calibration_info(image_path: str) -> Dict[str, Optional[float]]:
    """
    Intenta extraer información de calibración de una imagen.
    
    Args:
        image_path: Ruta a la imagen
    
    Returns:
        Dict con 'dpi' y 'scale_factor' si están disponibles
    """
    try:
        from PIL import Image
        
        img = Image.open(image_path)
        
        # Intentar obtener DPI de los metadatos
        dpi = None
        if hasattr(img, 'info') and 'dpi' in img.info:
            dpi_tuple = img.info['dpi']
            # Tomar el promedio si hay DPI diferente en X e Y
            dpi = (dpi_tuple[0] + dpi_tuple[1]) / 2 if isinstance(dpi_tuple, tuple) else dpi_tuple
        
        return {
            'dpi': dpi,
            'scale_factor': dpi / 2.54 if dpi else None
        }
    except Exception as e:
        print(f"⚠️  No se pudo leer la información de calibración: {e}")
        return {'dpi': None, 'scale_factor': None}


def estimate_foot_scale(foot_length_px: float, avg_foot_length_cm: float = 25.0) -> float:
    """
    Estima el factor de escala basándose en la longitud promedio de un pie.
    
    Args:
        foot_length_px: Longitud del pie en píxeles
        avg_foot_length_cm: Longitud promedio de un pie en cm (por defecto ~25 cm)
    
    Returns:
        float: Píxeles por centímetro estimado
    
    Note:
        La longitud promedio de un pie adulto es aproximadamente:
        - Mujer: 22-25 cm
        - Hombre: 25-28 cm
        Ajusta avg_foot_length_cm según tu caso de uso.
    """
    return foot_length_px / avg_foot_length_cm


# ============================================================================
# ESCALAS CONOCIDAS POR DATASET
# ============================================================================

# Escalas conocidas por dataset
# Nota: CAD tiene píxeles rectangulares (diferentes escalas en X e Y)

DATASET_INFO = {
    'stepup': {
        'description': 'StepUp-P150: 4 sensores/cm², 100 Hz',
        'scale_x_px_per_cm': 2.0,    # 2 px/cm en X
        'scale_y_px_per_cm': 2.0,    # 2 px/cm en Y (píxeles cuadrados)
        'cm_per_px_x': 0.5,          # 1 px = 0.5 cm en X
        'cm_per_px_y': 0.5,          # 1 px = 0.5 cm en Y
        'raw_data': True,            # Datos crudos, sin redimensionar al exportar
    },
    'cad': {
        'description': 'CAD Footscan: zooms=(7.62, 5.08, 2.0) mm, 500 Hz',
        'scale_x_px_per_cm': 1.31,   # ~1.31 px/cm en X (DATOS CRUDOS)
        'scale_y_px_per_cm': 1.97,   # ~1.97 px/cm en Y (DATOS CRUDOS)
        'cm_per_px_x': 0.762,        # 1 px = 0.762 cm en X (7.62 mm) - DATOS CRUDOS
        'cm_per_px_y': 0.508,        # 1 px = 0.508 cm en Y (5.08 mm) - DATOS CRUDOS
        'raw_data': False,           # Las imágenes exportadas están redimensionadas (DPI=150, figsize)
        'export_dpi': 150,           # DPI usado al exportar con plt.savefig
        'export_figsize': (5, 6),    # figsize usado al exportar (inches)
        'raw_size_approx': (34, 19), # Tamaño aproximado de datos crudos (H, W)
    },
    'casia': {
        'description': 'CASIA-D: Necesita verificación en CASgait.mat',
        'scale_x_px_per_cm': None,
        'scale_y_px_per_cm': None,
        'cm_per_px_x': None,
        'cm_per_px_y': None,
        'raw_data': True,            # Las imágenes se guardan sin redimensionar
    }
}


def detect_dataset_from_path(image_path: str) -> Optional[str]:
    """
    Intenta detectar el dataset de origen basándose en la ruta del archivo.
    
    Args:
        image_path: Ruta a la imagen
    
    Returns:
        str: 'stepup', 'cad', 'casia', o None si no se puede determinar
    """
    path_lower = str(image_path).lower()
    
    if 'stepup' in path_lower:
        return 'stepup'
    elif 'cad' in path_lower or 'footscan' in path_lower:
        return 'cad'
    elif 'casia' in path_lower or 'casgait' in path_lower:
        return 'casia'
    
    return None


def get_dataset_scale(dataset: str) -> Optional[Tuple[float, float]]:
    """
    Obtiene la escala conocida para un dataset específico.
    
    Args:
        dataset: Nombre del dataset ('stepup', 'cad', 'casia')
    
    Returns:
        Tuple[float, float]: (cm_per_px_x, cm_per_px_y), o None si no se conoce
    
    Note:
        Algunos datasets tienen píxeles rectangulares (diferentes escalas en X e Y).
        - StepUp: píxeles cuadrados (0.5 cm × 0.5 cm)
        - CAD: píxeles rectangulares (0.762 cm × 0.508 cm)
    """
    info = DATASET_INFO.get(dataset.lower())
    if info and info['cm_per_px_x'] is not None:
        return (info['cm_per_px_x'], info['cm_per_px_y'])
    return None


def convert_with_dataset_info(
    width_px: float,
    height_px: float,
    image_path: Optional[str] = None,
    dataset: Optional[str] = None,
    scale_factor: Optional[float] = None
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Convierte píxeles a cm intentando detectar automáticamente la escala.
    
    IMPORTANTE: Algunos datasets tienen píxeles rectangulares (diferentes escalas en X e Y).
    Esta función maneja correctamente ambos casos.
    
    Args:
        width_px: Ancho en píxeles
        height_px: Alto en píxeles
        image_path: Ruta de la imagen (para detectar dataset)
        dataset: Nombre del dataset ('stepup', 'cad', 'casia')
        scale_factor: Factor de escala manual (sobrescribe detección automática)
    
    Returns:
        Tuple[Optional[float], Optional[float], str]: 
            (ancho_cm, alto_cm, método_usado)
    """
    # 1. Si se proporciona scale_factor manual, usarlo
    if scale_factor is not None:
        width_cm, height_cm = pixels_to_cm(width_px, height_px, scale_factor=scale_factor)
        return width_cm, height_cm, "scale_factor manual"
    
    # 2. Si se proporciona dataset, usar su escala conocida
    if dataset:
        scales = get_dataset_scale(dataset)
        if scales:
            cm_per_px_x, cm_per_px_y = scales
            width_cm = width_px * cm_per_px_x
            height_cm = height_px * cm_per_px_y
            return width_cm, height_cm, f"dataset {dataset}"
    
    # 3. Si se proporciona ruta, intentar detectar dataset
    if image_path:
        detected = detect_dataset_from_path(image_path)
        if detected:
            scales = get_dataset_scale(detected)
            if scales:
                cm_per_px_x, cm_per_px_y = scales
                width_cm = width_px * cm_per_px_x
                height_cm = height_px * cm_per_px_y
                return width_cm, height_cm, f"detectado: {detected}"
        
        # 4. Intentar leer DPI de la imagen
        calib_info = get_calibration_info(image_path)
        if calib_info['scale_factor']:
            width_cm, height_cm = pixels_to_cm(width_px, height_px, scale_factor=calib_info['scale_factor'])
            return width_cm, height_cm, f"DPI imagen: {calib_info['dpi']:.0f}"
    
    # No se pudo determinar la escala
    return None, None, "No se pudo determinar la escala"


def calculate_foot_dimensions_from_pressure(
    img_array: np.ndarray,
    dataset: Optional[str] = None,
    image_path: Optional[str] = None,
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    Calcula las dimensiones reales del pie basándose SOLO en píxeles con presión > threshold.
    
    Esto da dimensiones más precisas que usar el rectángulo de toda la imagen,
    ya que excluye espacios en blanco/márgenes.
    
    Args:
        img_array: Array 2D de la imagen de presiones
        dataset: Nombre del dataset ('stepup', 'cad', 'casia')
        image_path: Ruta de la imagen (para detectar dataset automáticamente)
        threshold: Umbral mínimo de presión para considerar un píxel (default: 0)
    
    Returns:
        Dict con:
            - 'width_px': Ancho del pie en píxeles
            - 'height_px': Alto del pie en píxeles
            - 'width_cm': Ancho del pie en cm (si se puede calcular)
            - 'height_cm': Alto del pie en cm (si se puede calcular)
            - 'area_px': Área del pie en píxeles
            - 'area_cm2': Área del pie en cm² (si se puede calcular)
            - 'dataset': Dataset detectado
            - 'method': Método usado para conversión
    
    Example:
        >>> import numpy as np
        >>> from PIL import Image
        >>> # Cargar imagen
        >>> img = Image.open("mi_imagen.png").convert('L')
        >>> img_array = np.array(img)
        >>> # Calcular dimensiones
        >>> dims = calculate_foot_dimensions_from_pressure(
        ...     img_array, 
        ...     image_path="Salida_StepUp/mi_imagen.png"
        ... )
        >>> print(f"Pie: {dims['width_cm']:.2f} × {dims['height_cm']:.2f} cm")
    """
    try:
        import cv2
    except ImportError:
        return {'error': 'OpenCV (cv2) no está instalado'}
    
    # Asegurar que sea 2D
    if img_array.ndim == 3:
        img_array = np.mean(img_array, axis=2)
    
    img_array = np.asarray(img_array, dtype=np.float32)
    
    # Crear máscara binaria de píxeles con presión
    binary = (img_array > threshold).astype(np.uint8) * 255
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'error': 'No se encontraron contornos con presión > threshold'}
    
    # Encontrar el contorno más grande (el pie)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calcular rectángulo de área mínima
    rect = cv2.minAreaRect(largest_contour)
    (center_x, center_y), (width_px, height_px), angle = rect
    
    # El rectángulo puede estar rotado, tomar el mayor como ancho y menor como alto
    # para consistencia (asumiendo que el pie es más largo que ancho)
    if width_px < height_px:
        width_px, height_px = height_px, width_px
    
    # Calcular área en píxeles (contando solo píxeles con presión)
    area_px = float(np.sum(img_array > threshold))
    
    # Detectar dataset si no se proporcionó
    detected_dataset = dataset
    if not detected_dataset and image_path:
        detected_dataset = detect_dataset_from_path(image_path)
    
    result = {
        'width_px': float(width_px),
        'height_px': float(height_px),
        'area_px': area_px,
        'dataset': detected_dataset,
        'width_cm': None,
        'height_cm': None,
        'area_cm2': None,
        'method': 'No se pudo determinar escala'
    }
    
    # Intentar convertir a cm
    if detected_dataset:
        info = DATASET_INFO.get(detected_dataset.lower())
        if info and info['cm_per_px_x'] is not None:
            # Verificar si necesitamos ajustar por redimensionamiento al exportar
            if not info.get('raw_data', True) and 'raw_size_approx' in info:
                # Dataset CAD: las imágenes fueron redimensionadas al exportar con plt.savefig
                # El tamaño físico REAL no cambia, solo la resolución de la imagen
                raw_h, raw_w = info['raw_size_approx']
                current_h, current_w = img_array.shape
                
                # Tamaño físico real de los datos originales en cm
                real_width_cm = raw_w * info['cm_per_px_x']   # 19 * 0.762 = ~14.5 cm
                real_height_cm = raw_h * info['cm_per_px_y']  # 34 * 0.508 = ~17.3 cm
                
                # Ahora, en la imagen exportada, ese mismo tamaño físico se representa
                # con más píxeles. Calcular cuántos cm representa cada píxel en la exportada:
                cm_per_px_x = real_width_cm / current_w   # 14.5 cm / 854 px
                cm_per_px_y = real_height_cm / current_h  # 17.3 cm / 461 px
                
                result['width_cm'] = width_px * cm_per_px_x
                result['height_cm'] = height_px * cm_per_px_y
                result['area_cm2'] = area_px * cm_per_px_x * cm_per_px_y
                result['method'] = f'dataset {detected_dataset} (ajustado por export DPI)'
            else:
                # Datos crudos o sin redimensionamiento
                cm_per_px_x, cm_per_px_y = info['cm_per_px_x'], info['cm_per_px_y']
                result['width_cm'] = width_px * cm_per_px_x
                result['height_cm'] = height_px * cm_per_px_y
                result['area_cm2'] = area_px * cm_per_px_x * cm_per_px_y
                result['method'] = f'dataset {detected_dataset}'
    
    return result