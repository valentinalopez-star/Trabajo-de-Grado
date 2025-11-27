#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de normalización y mejora de imágenes.

Este módulo proporciona funciones para ajustar el brillo, contraste y tamaño
de imágenes plantares. Es utilizado por main.py en la opción 4 (Analizar imagen individual)
para mejorar la visualización de las imágenes antes de calcular el CoP.

Funciones principales:
    - normalize_image: Ajusta brillo, contraste y tamaño de una imagen

Nota: En el flujo actual de main.py, se llama con output_size=None, lo que significa
que NO se redimensiona la imagen, solo se ajustan brillo y contraste.
"""

from PIL import Image, ImageEnhance
from typing import Optional, Tuple


def normalize_image(
    img: Image.Image,
    output_size: Optional[Tuple[int, int]] = None,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0
) -> Image.Image:
    """
    Normaliza una imagen ajustando brillo, contraste y opcionalmente el tamaño.
    
    Esta función se utiliza para mejorar la visualización de imágenes plantares
    sin alterar las dimensiones físicas de los datos (cuando output_size=None).
    
    Args:
        img: Imagen PIL a procesar
        output_size: Tamaño de salida (ancho, alto) en píxeles. Si es None, 
                    mantiene el tamaño original.
        brightness_factor: Factor de brillo (1.0 = sin cambios, >1.0 = más brillante,
                          <1.0 = más oscuro). Debe ser > 0.
        contrast_factor: Factor de contraste (1.0 = sin cambios, >1.0 = más contraste,
                        <1.0 = menos contraste). Debe ser > 0.
    
    Returns:
        Image.Image: Imagen procesada con los ajustes aplicados
    
    Raises:
        ValueError: Si brightness_factor o contrast_factor son <= 0
    
    Examples:
        >>> from PIL import Image
        >>> img = Image.open("foot.png")
        >>> # Aumentar brillo y contraste sin cambiar tamaño
        >>> img_mejorada = normalize_image(img, brightness_factor=1.2, contrast_factor=1.3)
        >>> # Redimensionar a 512x512 con ajustes
        >>> img_redim = normalize_image(img, output_size=(512, 512), brightness_factor=1.1)
    
    Note:
        - En main.py se usa con output_size=None para preservar las dimensiones originales
        - Los factores de brillo y contraste típicos son 1.2 y 1.3 respectivamente
        - El redimensionamiento usa LANCZOS para mejor calidad
    """
    # Validación de parámetros
    if brightness_factor <= 0 or contrast_factor <= 0:
        raise ValueError("brightness_factor y contrast_factor deben ser > 0.")

    # Crear copia para no modificar la imagen original
    out = img.copy()

    # Ajustar brillo si es necesario
    if brightness_factor != 1.0:
        enhancer = ImageEnhance.Brightness(out)
        out = enhancer.enhance(brightness_factor)

    # Ajustar contraste si es necesario
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(out)
        out = enhancer.enhance(contrast_factor)

    # Redimensionar solo si se especifica un tamaño de salida
    if output_size is not None:
        # Usar LANCZOS para mejor calidad de redimensionamiento
        try:
            # PIL >= 9.1.0
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            # PIL < 9.1.0
            resample = Image.LANCZOS
        out = out.resize(output_size, resample=resample)

    return out