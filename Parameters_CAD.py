#!/usr/bin/env python3
"""
Cálculo de parámetros biomecánicos (CoF y CoP) para el CADDataset.

DESCRIPCIÓN:
    Este módulo analiza archivos .nii del CADDataset (sistema Footscan) y calcula
    métricas biomecánicas de distribución de fuerzas y centro de presión durante
    la fase de apoyo (stance phase) de la marcha.

MÉTRICAS CALCULADAS:

    1. CoF (Center of Force - Centro de Fuerza):
       - Distribución PROMEDIO de la fuerza en cada región del pie
       - Se calcula para cada frame y se promedia sobre toda la pisada
       - Indica qué porcentaje de la fuerza total actúa en cada región
       - Fórmula: CoF_región = Σ(presión_región) / Σ(presión_total) × 100%
    
    2. CoP (Center of Pressure - Centro de Presión):
       - Porcentaje de FRAMES en que el CoP cae dentro de cada región
       - El CoP es el punto donde se concentra el peso en cada instante
       - Indica en qué región del pie pasa más tiempo el centro de presión
       - Fórmula: CoP_región = (frames_en_región / frames_totales) × 100%

REGIONES DEL PIE:
    El pie se divide en 4 regiones anatómicas según DEFAULT_RATIOS (0.30, 0.55, 0.85):
    - Retropié (0-30%): Talón
    - Mediopié (30-55%): Arco plantar
    - Antepié (55-85%): Metatarsos
    - Dedos (85-100%): Falanges

ARCHIVOS CSV GENERADOS:

    1. Footsteps_detail.csv:
       - Una fila por cada archivo .nii procesado
       - Incluye: nombre archivo, lado (left/right), número de trial
       - Métricas: stance time, frames válidos, CoF por región, CoP por región
       - Formato: Separador ';', decimales con ',' (estilo europeo)
    
    2. Summary.csv:
       - Tres filas de resumen: Left, Right, Overall
       - Promedios de todas las métricas por lado
       - Mismo formato que Footsteps_detail.csv

USO EN EL PROYECTO:
    - main.py Opción 6: "Calcular parámetros para paciente CAD"
    - Procesa todos los archivos .nii de un paciente (ej: C01, HV01)
    - Guarda resultados en Salida_CAD/{paciente_id}/

FRECUENCIAS DE MUESTREO:
    - Pacientes clínicos (C01-C10): 500 fps (2 ms/frame)
    - Voluntarios Hallux Valgus (HV01-HV05): 200 Hz
    - Se configura automáticamente en main.py según el ID del paciente

FUNCIONES PRINCIPALES:
    - export_parameters(): Procesa lista de archivos .nii y genera CSVs
    - export_parameters_from_directory(): Procesa directorio completo
    - _analyze_nii_file(): Analiza un archivo .nii individual
    - calculate_cof(): Calcula distribución de fuerza por región
    - calculate_cop(): Calcula distribución de frames del CoP por región

PROCESO DE ANÁLISIS:
    1. Carga archivo .nii con nibabel
    2. Endereza frames usando straighten_pressure_frames()
    3. Detecta orientación del pie (horizontal/vertical, talón arriba/abajo)
    4. Crea máscaras de regiones según DEFAULT_RATIOS
    5. Calcula CoF y CoP para cada región
    6. Guarda resultados en CSVs con formato europeo

NOTA IMPORTANTE:
    - Los frames se enderezan ANTES de calcular CoF y CoP para consistencia
    - effective_invert=False para CAD (frames ya orientados correctamente)
    - El parsing de nombres de archivo extrae automáticamente lado y número de trial
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import nibabel as nib
import numpy as np

from Rotate import straighten_pressure_frames

# Regiones del pie
REGIONS: Tuple[str, ...] = ("retropie", "mediopie", "antepie", "dedos")
DEFAULT_RATIOS: Tuple[float, float, float] = (0.30, 0.55, 0.85)

# Campos para el CSV de detalle
DETAIL_FIELDS: List[str] = [
    "File name",
    "Side",
    "Trial number",
    "Stance time (s)",
    "Num. frames valid",
    "Mean CoF retropie (%)",
    "Mean CoF mediopie (%)",
    "Mean CoF antepie (%)",
    "Mean CoF dedos (%)",
    "Frames CoP retropie (%)",
    "Frames CoP mediopie (%)",
    "Frames CoP antepie (%)",
    "Frames CoP dedos (%)",
]


# --------------------------------------------------------------------------- #
# Utilidades internas
# --------------------------------------------------------------------------- #

def _compute_orientation(img: np.ndarray) -> Tuple[bool, bool]:
    """
    Determina si el eje longitudinal es horizontal (True) o vertical y si hay
    que invertir el eje (talón a la derecha o arriba).
    """
    h, w = img.shape[:2]
    mask = (img > 0).astype(np.uint8)
    is_horizontal = w >= h

    if is_horizontal:
        width = max(1, int(round(w * 0.2)))
        left = mask[:, :width].sum()
        right = mask[:, w - width :].sum()
        invert = right > left  # talón hacia la derecha
    else:
        height = max(1, int(round(h * 0.2)))
        top = mask[:height, :].sum()
        bottom = mask[h - height :, :].sum()
        invert = top > bottom  # talón hacia abajo

    return is_horizontal, invert


def _region_masks(
    shape: Tuple[int, int], is_horizontal: bool, invert: bool, ratios: Sequence[float]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Retorna máscaras booleanas por región y la imagen de coordenadas normalizada.
    """
    h, w = shape
    ratios = tuple(ratios)

    if is_horizontal:
        axis = max(1.0, w - 1.0)
        coord = np.arange(w, dtype=float)
        norm = (axis - coord) / axis if invert else coord / axis
        norm_img = np.broadcast_to(norm, (h, w))
    else:
        axis = max(1.0, h - 1.0)
        coord = np.arange(h, dtype=float)
        norm = (axis - coord) / axis if invert else coord / axis
        norm_img = np.broadcast_to(norm[:, None], (h, w))

    masks = {
        "retropie": norm_img < ratios[0],
        "mediopie": (norm_img >= ratios[0]) & (norm_img < ratios[1]),
        "antepie": (norm_img >= ratios[1]) & (norm_img < ratios[2]),
        "dedos": norm_img >= ratios[2],
    }
    return masks, norm_img


def calculate_cof(
    roi_stack: Sequence[np.ndarray], masks: Mapping[str, np.ndarray]
) -> Tuple[Dict[str, float], int]:
    """
    Calcula el porcentaje medio de fuerza (CoF) por región y devuelve también
    el número de frames con fuerza válida.
    """
    pct_acc = {region: [] for region in REGIONS}
    valid_frames = 0

    for img in roi_stack:
        total_force = float(img.sum())
        if total_force <= 0:
            continue
        valid_frames += 1
        for region in REGIONS:
            mask = masks[region]
            value = float(img[mask].sum()) if mask.any() else 0.0
            pct_acc[region].append(value / total_force if total_force > 0 else 0.0)

    mean_pct = {
        region: float(np.mean(pct_acc[region])) if pct_acc[region] else 0.0
        for region in REGIONS
    }
    return mean_pct, valid_frames


def calculate_cop(
    roi_stack: Sequence[np.ndarray],
    norm_img: np.ndarray,
    is_horizontal: bool,
    ratios: Sequence[float],
) -> Dict[str, float]:
    """
    Calcula el porcentaje de frames cuyo CoP cae dentro de cada región.
    """
    counts = {region: 0 for region in REGIONS}
    valid = 0

    for img in roi_stack:
        total_force = float(img.sum())
        if total_force <= 0:
            continue
        cop_x, cop_y = _cop(img)
        region = _cop_region(cop_x, cop_y, norm_img, is_horizontal, ratios)
        if region is None:
            continue
        counts[region] += 1
        valid += 1

    return {
        region: (counts[region] / valid if valid > 0 else 0.0) for region in REGIONS
    }


def _cop(img: np.ndarray) -> Tuple[float, float]:
    """Calcula el Centro de Presión (CoP) de una imagen."""
    total = float(img.sum())
    if total <= 0:
        return float("nan"), float("nan")
    y_idx, x_idx = np.indices(img.shape)
    cop_x = float((img * x_idx).sum() / total)
    cop_y = float((img * y_idx).sum() / total)
    return cop_x, cop_y


def _cop_region(
    cop_x: float,
    cop_y: float,
    norm_img: np.ndarray,
    is_horizontal: bool,
    ratios: Sequence[float],
) -> str | None:
    """Determina en qué región del pie está el CoP."""
    if not np.isfinite(cop_x) or not np.isfinite(cop_y):
        return None

    h, w = norm_img.shape
    if is_horizontal:
        idx = int(round(cop_x))
        if not (0 <= idx < w):
            return None
        value = float(norm_img[0, idx])
    else:
        idx = int(round(cop_y))
        if not (0 <= idx < h):
            return None
        value = float(norm_img[idx, 0])

    if value < ratios[0]:
        return "retropie"
    if value < ratios[1]:
        return "mediopie"
    if value < ratios[2]:
        return "antepie"
    return "dedos"


# --------------------------------------------------------------------------- #
# Procesamiento de archivo .nii
# --------------------------------------------------------------------------- #

def _analyze_nii_file(
    nii_file_path: Path,
    fps: float,
    ratios: Sequence[float],
) -> Dict[str, float | int | str]:
    """
    Analiza un archivo .nii y calcula CoP y CoF.

    Args:
        nii_file_path: Ruta al archivo .nii
        fps: Frames por segundo
        ratios: Ratios para dividir el pie en regiones

    Returns:
        Diccionario con todas las métricas calculadas
    """
    # Cargar archivo .nii
    img = nib.load(str(nii_file_path))
    data = img.get_fdata()

    # Extraer roi_stack (cada frame es una imagen 2D) y enderezar
    roi_stack_raw = [data[:, :, i] for i in range(data.shape[2])]
    straightened = straighten_pressure_frames(roi_stack_raw, nombre_db="CAD")
    roi_stack = straightened.frames

    roi_ref = roi_stack[len(roi_stack) // 2]
    is_horizontal, invert = _compute_orientation(roi_ref)
    
    # Para CAD: NO invertir físicamente, NO usar inversión lógica
    # Los frames ya están bien orientados, las máscaras deben ser directas
    effective_invert = False

    # Crear máscaras de regiones sin inversión (0-30% abajo = retropié)
    masks, norm_img = _region_masks(
        roi_ref.shape, is_horizontal, effective_invert, ratios
    )

    # Calcular tiempo de stance
    stance_time = len(roi_stack) / float(fps)

    # Calcular CoF y CoP
    mean_pct, valid_frames = calculate_cof(roi_stack, masks)
    pct_cop = calculate_cop(roi_stack, norm_img, is_horizontal, ratios)

    return {
        "stance_time_s": float(stance_time),
        "n_frames_valid": int(valid_frames),
        "mean_pct": mean_pct,
        "pct_cop": pct_cop,
    }


def _parse_filename(filename: str) -> Tuple[str, int]:
    """
    Extrae el lado (left/right) y número de trial del nombre del archivo.

    Ejemplo: "left_foot_trial_01.nii" -> ("left", 1)
    """
    name = Path(filename).stem
    parts = name.split("_")

    # Buscar lado
    side = "unknown"
    if "left" in name.lower():
        side = "left"
    elif "right" in name.lower():
        side = "right"

    # Buscar número de trial
    trial_number = 0
    for part in parts:
        if part.isdigit():
            trial_number = int(part)
            break

    return side, trial_number


# --------------------------------------------------------------------------- #
# Export principal
# --------------------------------------------------------------------------- #

def export_parameters(
    nii_files: List[Path | str],
    output_dir: Path | str = "Salida",
    fps: float = 500.0,  # 2ms por frame = 500 fps para CADDataset
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> Tuple[Path, Path]:
    """
    Calcula métricas de CoF (distribución de fuerza) y CoP (distribución de
    frames) para todos los archivos .nii proporcionados y guarda dos CSV.

    Args:
        nii_files: Lista de rutas a archivos .nii
        output_dir: Directorio de salida
        fps: Frames por segundo (500 fps = 2ms por frame según header CAD)
        ratios: Porcentajes (retropie, mediopie, antepie) para dividir el pie

    Returns:
        tuple(Path, Path): rutas a Footsteps_detail.csv y Summary.csv
    """
    output_dir = Path(output_dir)
    detail_path = output_dir / "Footsteps_detail.csv"
    summary_path = output_dir / "Summary.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: List[Dict[str, float | int | str]] = []

    print(f"\n{'='*80}")
    print(f"Procesando {len(nii_files)} archivos .nii")
    print(f"{'='*80}\n")

    for nii_file in nii_files:
        nii_path = Path(nii_file)

        if not nii_path.exists():
            print(f"⚠️  Archivo no encontrado: {nii_path}")
            continue

        print(f"Procesando: {nii_path.name}")

        try:
            # Analizar archivo
            analysis = _analyze_nii_file(nii_path, fps, ratios)

            # Extraer información del nombre del archivo
            side, trial_number = _parse_filename(nii_path.name)

            # Construir fila de detalle
            row = {
                "File name": nii_path.name,
                "Side": side,
                "Trial number": trial_number,
                "Stance time (s)": round(analysis["stance_time_s"], 6),
                "Num. frames valid": int(analysis["n_frames_valid"]),
            }

            # Agregar métricas CoF
            for region in REGIONS:
                row[f"Mean CoF {region} (%)"] = round(
                    analysis["mean_pct"][region] * 100, 6
                )

            # Agregar métricas CoP
            for region in REGIONS:
                row[f"Frames CoP {region} (%)"] = round(
                    analysis["pct_cop"][region] * 100, 6
                )

            detail_rows.append(row)
            print(f"  ✓ Stance: {analysis['stance_time_s']*1000:.1f} ms, "
                  f"Frames: {analysis['n_frames_valid']}")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue

    if not detail_rows:
        raise RuntimeError("No se pudo procesar ningún archivo .nii válido.")

    print(f"\n{'='*80}")
    print(f"Guardando resultados...")
    print(f"{'='*80}\n")

    # Guardar CSV de detalle
    _write_csv(detail_path, DETAIL_FIELDS, detail_rows)
    print(f"✓ Detalle guardado: {detail_path}")

    # Construir y guardar resumen
    summary_rows = _build_summary(detail_rows)
    _write_csv(summary_path, DETAIL_FIELDS, summary_rows)
    print(f"✓ Resumen guardado: {summary_path}")

    return detail_path, summary_path


def _format_number(value: float | int | str) -> str:
    """
    Formatea números al estilo europeo/latinoamericano:
    - Separador de decimales: coma (,)
    - Separador de miles: punto (.)
    """
    if isinstance(value, str):
        return value

    if isinstance(value, int):
        # Para enteros: agregar separador de miles
        return f"{value:,.0f}".replace(",", ".")

    if isinstance(value, float):
        # Para flotantes: 6 decimales con coma como separador
        formatted = f"{value:,.6f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return formatted

    return str(value)


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Dict[str, object]]):
    """Escribe un archivo CSV con formato europeo."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(headers), delimiter=';')
        writer.writeheader()
        for row in rows:
            # Formatear valores numéricos al estilo europeo
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, (int, float)) and key not in ["Trial number", "Num. frames valid"]:
                    formatted_row[key] = _format_number(value)
                else:
                    formatted_row[key] = value
            writer.writerow(formatted_row)


def _build_summary(
    detail_rows: Sequence[Dict[str, float | int | str]]
) -> List[Dict[str, float | int | str]]:
    """
    Construye filas de resumen agregando por lado (Left, Right, Overall).
    """
    def select_by_side(target: str) -> List[Dict[str, float | int | str]]:
        return [row for row in detail_rows if str(row["Side"]).lower() == target]

    summaries: List[Dict[str, float | int | str]] = []

    for label, selector in (
        ("left", lambda: select_by_side("left")),
        ("right", lambda: select_by_side("right")),
        ("overall", lambda: list(detail_rows)),
    ):
        rows = selector()
        if not rows:
            continue
        summary = _aggregate_rows(rows, label)
        summaries.append(summary)

    return summaries


def _aggregate_rows(
    rows: Sequence[Dict[str, float | int | str]], side_label: str
) -> Dict[str, float | int | str]:
    """
    Agrega múltiples filas en una sola fila de resumen.
    """
    def mean(field: str) -> float:
        values = [float(row[field]) for row in rows if field in row]
        return float(np.mean(values)) if values else 0.0

    def sum_int(field: str) -> int:
        values = [int(row[field]) for row in rows if field in row]
        return int(sum(values))

    base = rows[0]
    aggregated: Dict[str, float | int | str] = {
        "File name": "Summary",
        "Side": side_label.capitalize(),
        "Trial number": "All",
        "Stance time (s)": round(mean("Stance time (s)"), 6),
        "Num. frames valid": sum_int("Num. frames valid"),
    }

    # Promediar métricas CoF
    for region in REGIONS:
        aggregated[f"Mean CoF {region} (%)"] = round(
            mean(f"Mean CoF {region} (%)"), 6
        )

    # Promediar métricas CoP
    for region in REGIONS:
        aggregated[f"Frames CoP {region} (%)"] = round(
            mean(f"Frames CoP {region} (%)"), 6
        )

    return aggregated


# --------------------------------------------------------------------------- #
# Utilidades adicionales
# --------------------------------------------------------------------------- #

def export_parameters_from_directory(
    directory: Path | str,
    output_dir: Path | str = "Salida",
    pattern: str = "*.nii",
    fps: float = 500.0,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> Tuple[Path, Path]:
    """
    Procesa todos los archivos .nii de un directorio.

    Args:
        directory: Directorio con archivos .nii
        output_dir: Directorio de salida
        pattern: Patrón de archivos a buscar (por defecto "*.nii")
        fps: Frames por segundo
        ratios: Ratios para dividir el pie en regiones

    Returns:
        tuple(Path, Path): rutas a Footsteps_detail.csv y Summary.csv
    """
    directory = Path(directory)
    nii_files = sorted(directory.glob(pattern))

    if not nii_files:
        raise RuntimeError(f"No se encontraron archivos .nii en {directory}")

    print(f"\nEncontrados {len(nii_files)} archivos .nii en {directory}")

    return export_parameters(nii_files, output_dir, fps, ratios)


def display_file_info(nii_file: Path | str) -> None:
    """
    Muestra información detallada de un archivo .nii.

    Args:
        nii_file: Ruta al archivo .nii
    """
    print(f"\n{'='*80}")
    print(f"Información del archivo: {Path(nii_file).name}")
    print(f"{'='*80}")

    img = nib.load(str(nii_file))
    data = img.get_fdata()
    header = img.header

    print(f"\n📊 Información básica:")
    print(f"  - Dimensiones: {data.shape}")
    print(f"  - Tipo de datos: {data.dtype}")
    print(f"  - Tamaño en memoria: {data.nbytes / (1024*1024):.2f} MB")

    print(f"\n📈 Estadísticas de los datos:")
    print(f"  - Valor mínimo: {np.min(data):.4f}")
    print(f"  - Valor máximo: {np.max(data):.4f}")
    print(f"  - Valor promedio: {np.mean(data):.4f}")
    print(f"  - Desviación estándar: {np.std(data):.4f}")

    print(f"\n🔧 Información del header:")
    print(f"  - Dimensiones del header: {header['dim']}")
    print(f"  - Pixdim (resolución): {header['pixdim']}")

    if len(data.shape) == 3:
        print(f"\n📦 Volumen 3D detectado:")
        print(f"  - Frames/slices: {data.shape[2]}")
        non_zero_frames = np.sum(np.any(data > 0, axis=(0, 1)))
        print(f"  - Frames con datos (valor > 0): {non_zero_frames}")
