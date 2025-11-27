

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo del Centro de Presión (CoP) para imágenes estáticas de presiones plantares.

DESCRIPCIÓN:
    Este módulo calcula el Centro de Presión (CoP) de una imagen estática que 
    representa la distribución de presiones de un pie. El CoP es el punto donde
    se concentra el peso del cuerpo y se calcula como el centroide ponderado
    por la intensidad de presión de cada píxel.

FÓRMULA DEL CoP:
    CoP_x = Σ(x * P(x,y)) / Σ(P(x,y))
    CoP_y = Σ(y * P(x,y)) / Σ(P(x,y))
    
    donde P(x,y) es la presión (intensidad) en el píxel (x,y).

USO EN EL PROYECTO:
    - Opción 4 de main.py: Analizar una imagen individual
    - Puede usarse standalone para procesar imágenes de cualquier dataset:
        * CASIA (Opción 1 del main.py)
        * CAD (Opción 2 del main.py)
        * StepUp (Opción 3 del main.py)

FUNCIONES PRINCIPALES:
    - compute_cop_static(): Calcula el CoP de un array numpy
    - load_image_as_array(): Carga una imagen y la convierte a array
    - visualize_cop(): Crea visualización con el CoP marcado
    - process_static_image(): Función completa que procesa y visualiza

NOTA:
    Este módulo calcula CoP para imágenes ESTÁTICAS (peak de presión).
    Para secuencias temporales (frames), ver Parameters_StepUp.py y Parameters_CAD.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image


def compute_cop_static(img: np.ndarray) -> Tuple[float, float]:
    """
    Calcula el Centro de Presión (CoP) de una imagen estática de presiones.
    
    El CoP se calcula como el centroide ponderado por intensidad:
        CoP_x = Σ(x * I(x,y)) / Σ(I(x,y))
        CoP_y = Σ(y * I(x,y)) / Σ(I(x,y))
    
    donde I(x,y) es la intensidad (presión) en el píxel (x,y).
    
    Args:
        img: Array 2D numpy con la imagen de presiones (H, W)
    
    Returns:
        Tuple[float, float]: Coordenadas (cop_x, cop_y) del CoP.
            Si la imagen no tiene presión válida, retorna (nan, nan).
    
    Examples:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100) * 100
        >>> cop_x, cop_y = compute_cop_static(img)
        >>> isinstance(cop_x, float) and isinstance(cop_y, float)
        True
        >>> 0 <= cop_x < 100 and 0 <= cop_y < 100
        True
    """
    img = np.asarray(img, dtype=np.float64)
    
    # Verificar que la imagen sea 2D
    if img.ndim != 2:
        raise ValueError(f"La imagen debe ser 2D, pero tiene forma {img.shape}")
    
    total = float(img.sum())
    if total <= 0:
        return float("nan"), float("nan")
    
    # Crear índices de coordenadas
    y_idx, x_idx = np.indices(img.shape, dtype=np.float64)
    
    # Calcular centroide ponderado
    cop_x = float((img * x_idx).sum() / total)
    cop_y = float((img * y_idx).sum() / total)
    
    return cop_x, cop_y


def load_image_as_array(image_path: Path | str) -> np.ndarray:
    """
    Carga una imagen desde un archivo y la convierte a array numpy.
    
    Args:
        image_path: Ruta al archivo de imagen (PNG, JPG, etc.)
    
    Returns:
        Array 2D numpy con valores de intensidad normalizados a 0-255
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {image_path}")
    
    # Cargar imagen con PIL
    img = Image.open(image_path)
    
    # Convertir a escala de grises si es necesario
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convertir a array numpy
    img_array = np.array(img, dtype=np.float64)
    
    return img_array


def visualize_cop(
    img: np.ndarray,
    cop_x: float,
    cop_y: float,
    output_path: Path | str | None = None,
    title: str = "CoP en imagen estática",
) -> None:
    """
    Visualiza la imagen con el CoP marcado.
    
    Args:
        img: Array 2D con la imagen de presiones
        cop_x: Coordenada X del CoP
        cop_y: Coordenada Y del CoP
        output_path: Ruta donde guardar la figura (opcional)
        title: Título de la figura
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Crear colormap jet donde 0 es negro (como en otros scripts del proyecto)
    jet = mpl.colormaps["jet"](np.linspace(0, 1, 256))
    jet[0] = [0, 0, 0, 1]
    cmap_jet = mpl.colors.ListedColormap(jet)
    
    # Mostrar imagen
    im = ax.imshow(img, cmap=cmap_jet, origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Presión", fraction=0.046, pad=0.04)
    
    # Marcar el CoP si es válido
    if np.isfinite(cop_x) and np.isfinite(cop_y):
        ax.plot(
            cop_x,
            cop_y,
            "ro",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label=f"CoP: ({cop_x:.2f}, {cop_y:.2f})",
            zorder=5,
        )
        ax.legend(loc="best", fontsize=12)
    
    ax.set_xlabel("X (píxeles)", fontsize=12)
    ax.set_ylabel("Y (píxeles)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Figura guardada en: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def process_static_image(
    image_path: Path | str,
    visualize: bool = True,
    output_path: Path | str | None = None,
) -> Tuple[float, float]:
    """
    Procesa una imagen estática y calcula su CoP.
    
    Args:
        image_path: Ruta a la imagen estática de presiones
        visualize: Si True, muestra/guarda una visualización
        output_path: Ruta donde guardar la visualización (opcional)
    
    Returns:
        Tuple[float, float]: Coordenadas (cop_x, cop_y) del CoP
    """
    image_path = Path(image_path)
    
    print(f"\n📊 Procesando imagen: {image_path.name}")
    print(f"📁 Ruta completa: {image_path}")
    
    # Cargar imagen
    img = load_image_as_array(image_path)
    print(f"✓ Imagen cargada: {img.shape[0]}x{img.shape[1]} píxeles")
    
    # Calcular CoP
    cop_x, cop_y = compute_cop_static(img)
    
    if np.isfinite(cop_x) and np.isfinite(cop_y):
        print(f"\n✅ CoP calculado exitosamente:")
        print(f"   CoP_x = {cop_x:.2f} píxeles")
        print(f"   CoP_y = {cop_y:.2f} píxeles")
        print(f"   CoP relativo: ({cop_x/img.shape[1]:.2%}, {cop_y/img.shape[0]:.2%})")
    else:
        print("\n⚠️  No se pudo calcular el CoP (imagen sin presión válida)")
    
    # Visualizar si se solicita
    if visualize:
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_cop.png"
        
        visualize_cop(
            img,
            cop_x,
            cop_y,
            output_path=output_path,
            title=f"CoP - {image_path.name}",
        )
    
    return cop_x, cop_y


def main():
    """
    Función principal interactiva para calcular CoP de imágenes estáticas.
    Permite seleccionar una imagen mediante diálogo o usar argumentos de línea de comandos.
    """
    import sys
    import tkinter as tk
    from tkinter import filedialog
    
    # Si hay argumentos de línea de comandos, usar modo CLI
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        visualize = "--no-viz" not in sys.argv
        
        output_path = None
        if "--output" in sys.argv:
            idx = sys.argv.index("--output")
            if idx + 1 < len(sys.argv):
                output_path = sys.argv[idx + 1]
        
        try:
            cop_x, cop_y = process_static_image(
                image_path,
                visualize=visualize,
                output_path=output_path,
            )
            
            if np.isfinite(cop_x) and np.isfinite(cop_y):
                print(f"\n📊 Resultado final:")
                print(f"   CoP: ({cop_x:.2f}, {cop_y:.2f})")
        except Exception as e:
            print(f"\n❌ Error al procesar: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return
    
    # Modo interactivo: mostrar diálogo para seleccionar imagen
    print("\n" + "=" * 60)
    print("   CALCULADOR DE CoP PARA IMÁGENES ESTÁTICAS")
    print("=" * 60)
    print("\nSelecciona una imagen de peak de presiones...")
    
    root = tk.Tk()
    root.withdraw()
    
    image_path = filedialog.askopenfilename(
        title="Selecciona una imagen de presiones",
        filetypes=[
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("Todos los archivos", "*.*")
        ]
    )
    
    if not image_path:
        print("❌ No se seleccionó ninguna imagen. Saliendo...")
        return
    
    image_path = Path(image_path)
    
    # Preguntar si quiere visualización
    print("\n" + "-" * 60)
    generar_viz = input("¿Desea generar una visualización con el CoP marcado? (s/n): ").strip().lower()
    visualize = generar_viz in ['s', 'si', 'sí', 'yes', 'y']
    
    output_path = None
    if visualize:
        print("\n" + "-" * 60)
        guardar = input("¿Desea guardar la visualización en un archivo? (s/n): ").strip().lower()
        if guardar in ['s', 'si', 'sí', 'yes', 'y']:
            output_path = filedialog.asksaveasfilename(
                title="Guardar visualización como...",
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg"),
                    ("Todos los archivos", "*.*")
                ],
                initialfile=f"{image_path.stem}_cop.png"
            )
            if not output_path:
                output_path = None
    
    try:
        cop_x, cop_y = process_static_image(
            image_path,
            visualize=visualize,
            output_path=output_path,
        )
        
        if np.isfinite(cop_x) and np.isfinite(cop_y):
            print("\n" + "=" * 60)
            print("📊 RESULTADO FINAL")
            print("=" * 60)
            print(f"   Archivo: {image_path.name}")
            print(f"   CoP absoluto: ({cop_x:.2f}, {cop_y:.2f}) píxeles")
            
            # Cargar imagen para obtener dimensiones
            img = load_image_as_array(image_path)
            print(f"   CoP relativo: ({cop_x/img.shape[1]:.2%}, {cop_y/img.shape[0]:.2%})")
            print(f"   Dimensiones imagen: {img.shape[1]}x{img.shape[0]} píxeles")
            
            if visualize and output_path:
                print(f"\n✅ Visualización guardada en: {output_path}")
            elif visualize:
                print(f"\n✅ Visualización mostrada en ventana")
            
            print("=" * 60)
        else:
            print("\n⚠️  No se pudo calcular el CoP (imagen sin presión válida)")
            
    except Exception as e:
        print(f"\n❌ Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()
