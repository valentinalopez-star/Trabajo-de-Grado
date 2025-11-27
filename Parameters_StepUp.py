#!/usr/bin/env python3
"""
Cálculo de parámetros biomecánicos (CoF y CoP) para el StepUpDataset.

DESCRIPCIÓN:
    Este módulo analiza datos del StepUpDataset (pasarela de presión StepUp-P150)
    y calcula métricas biomecánicas de distribución de fuerzas y centro de presión
    durante la fase de apoyo de la marcha.

DATASET STEPUP-P150:
    - Pasarela de presión: 3.6 m × 1.2 m
    - Densidad: 4 sensores/cm² (2 px/cm)
    - Frecuencia: 100 Hz
    - Formato: Archivos .npz con datos de trials completos
    - Estructura: participant_id / shoe_type / walk_condition

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
       - Una fila por cada pisada válida (Standing=0, Incomplete=0)
       - Incluye: participant_id, shoe_type, walk_condition, PassID, FootstepID, Side
       - Métricas: stance time, frames válidos, CoF por región, CoP por región
       - Formato: Separador ';', decimales con ',' (estilo europeo)
    
    2. Summary.csv:
       - Tres filas de resumen: Left, Right, Overall
       - Promedios de todas las métricas por lado
       - Mismo formato que Footsteps_detail.csv

USO EN EL PROYECTO:
    - main.py Opción 5: "Calcular parámetros para paciente StepUp"
    - Procesa todas las pisadas válidas de un participante/condición
    - Guarda resultados en Salida_StepUp/Participante_{id}/BF/W1/

FUNCIONES PRINCIPALES:
    - export_parameters(): Procesa todas las pisadas y genera CSVs
    - _analyze_footstep(): Analiza una pisada individual
    - _extract_roi_stack(): Extrae y endereza frames de una pisada
    - calculate_cof(): Calcula distribución de fuerza por región
    - calculate_cop(): Calcula distribución de frames del CoP por región

PROCESO DE ANÁLISIS:
    1. Carga metadata y trial del participante
    2. Filtra pisadas válidas (no standing, no incomplete)
    3. Para cada pisada:
       a. Extrae ROI y rota 90° (np.rot90)
       b. Endereza frames usando straighten_pressure_frames()
       c. Detecta orientación y aplica inversión física si es necesario
       d. Crea máscaras de regiones
       e. Calcula CoF y CoP
    4. Genera CSVs con formato europeo

DIFERENCIAS CON PARAMETERS_CAD:
    - StepUp usa metadata de pandas con filtros (Standing, Incomplete)
    - StepUp aplica np.rot90 inicial antes de enderezar
    - StepUp aplica inversión física cuando invert=True
    - CAD procesa archivos .nii directamente sin metadata

NOTA IMPORTANTE:
    - Los frames se enderezan ANTES de calcular CoF y CoP
    - effective_invert=False después de inversión física (talón abajo, dedos arriba)
    - El módulo NO calcula CPEI (ver Extras/compute_cpei.py para eso)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from StepUpDataset.utils import load_metadata, load_trial
from Rotate import straighten_pressure_frames

REGIONS: Tuple[str, ...] = ("retropie", "mediopie", "antepie", "dedos")
DEFAULT_RATIOS: Tuple[float, float, float] = (0.30, 0.55, 0.85)
DETAIL_FIELDS: List[str] = [
    "Participant id",
    "Shoe type",
    "Walk condition",
    "iPass",
    "FootstepID",
    "Side",
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


def _get_fps(step_row, fallback: float) -> float:
    for key in ("FPS", "FrameRate", "fps"):
        if key in step_row and np.isfinite(step_row[key]):
            value = float(step_row[key])
            if value > 0:
                return value
    return float(fallback)


def _extract_roi_stack(
    trial: np.ndarray,
    step_row,
    *,
    apply_inversion: bool,
) -> Tuple[List[np.ndarray], int, int, bool, bool]:
    """
    Extrae el conjunto de ROIs (rotadas) para un footstep y alinea orientación.

    Returns:
        roi_stack: Lista de imágenes (frame a frame) rotadas 90° para que el
            eje longitudinal quede horizontal cuando corresponda.
        start_frame: Frame inicial del footstep.
        end_frame: Frame final del footstep.
        is_horizontal: True si el eje longitudinal es horizontal.
        effective_invert: True si se requiere invertir la normalización para
            conservar talón→punta tras la extracción.
    """
    start_f = int(step_row["StartFrame"])
    end_f = int(step_row["EndFrame"])
    y0, y1 = int(step_row["Ymin"]), int(step_row["Ymax"])
    x0, x1 = int(step_row["Xmin"]), int(step_row["Xmax"])
    
    # Validar que los frames estén dentro del rango válido del trial
    max_frame = trial.shape[0] - 1
    start_f = max(0, min(start_f, max_frame))
    end_f = max(start_f, min(end_f, max_frame))

    roi_stack_raw = [
        np.rot90(trial[f, y0:y1, x0:x1]) for f in range(start_f, end_f + 1)
    ]
    if not roi_stack_raw:
        raise RuntimeError("El footstep no contiene frames válidos.")

    straightened = straighten_pressure_frames(roi_stack_raw, nombre_db="STEPUP")
    roi_stack = straightened.frames

    roi_ref = roi_stack[len(roi_stack) // 2]
    is_horizontal, invert = _compute_orientation(roi_ref)
    
    # Aplicar inversión física SOLO si invert=True (talón está hacia derecha/arriba)
    # Esto pone el talón abajo y los dedos arriba
    if invert:
        transform = np.fliplr if is_horizontal else np.flipud
        roi_stack = [transform(img) for img in roi_stack]
        roi_ref = roi_stack[len(roi_stack) // 2]
        is_horizontal, _ = _compute_orientation(roi_ref)
    
    # effective_invert siempre False (máscaras directas después de inversión física)
    effective_invert = False

    return roi_stack, start_f, end_f, is_horizontal, effective_invert


# --------------------------------------------------------------------------- #
# Procesamiento por footstep
# --------------------------------------------------------------------------- #

def _analyze_footstep(
    trial: np.ndarray,
    step_row,
    fps: float,
    ratios: Sequence[float],
) -> Dict[str, float | int | str]:
    (
        roi_stack,
        start_f,
        end_f,
        is_horizontal,
        effective_invert,
    ) = _extract_roi_stack(trial, step_row, apply_inversion=True)  # Invertir físicamente
    roi_ref = roi_stack[len(roi_stack) // 2]
    # effective_invert será False después de inversión física
    masks, norm_img = _region_masks(
        roi_ref.shape, is_horizontal, effective_invert, ratios
    )

    stance_time = (end_f - start_f + 1) / float(fps)
    mean_pct, valid_frames = calculate_cof(roi_stack, masks)
    pct_cop = calculate_cop(roi_stack, norm_img, is_horizontal, ratios)

    return {
        "stance_time_s": float(stance_time),
        "n_frames_valid": int(valid_frames),
        "mean_pct": mean_pct,
        "pct_cop": pct_cop,
    }


# --------------------------------------------------------------------------- #
# Export principal
# --------------------------------------------------------------------------- #

def export_parameters(
    participant_id: int,
    shoe_type: str,
    walk_condition: str,
    output_dir: Path | str = "Salida",
    fps_fallback: float = 100.0,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> Tuple[Path, Path]:
    """
    Calcula métricas de CoF (distribución de fuerza) y CoP (distribución de
    frames) para todas las pisadas válidas del participante y guarda dos CSV.

    Returns:
        tuple(Path, Path): rutas a Footsteps_detail.csv y Summary.csv
    """
    output_dir = Path(output_dir)
    detail_path = output_dir / "Footsteps_detail.csv"
    summary_path = output_dir / "Summary.csv"

    metadata = load_metadata(participant_id, shoe_type, walk_condition)
    trial = load_trial(participant_id, shoe_type, walk_condition)

    mask = (metadata["Standing"] == 0) & (metadata["Incomplete"] == 0)
    steps = metadata[mask].sort_values(["PassID", "FootstepID"])
    if steps.empty:
        raise RuntimeError("No se encontraron footsteps válidos con los filtros dados.")

    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: List[Dict[str, float | int | str]] = []
    for _, step in steps.iterrows():
        fps = _get_fps(step, fps_fallback)
        analysis = _analyze_footstep(trial, step, fps, ratios)
        row = {
            "Participant id": participant_id,
            "Shoe type": shoe_type,
            "Walk condition": walk_condition,
            "iPass": int(step["PassID"]),
            "FootstepID": int(step["FootstepID"]),
            "Side": str(step["Side"]),
            "Stance time (s)": round(analysis["stance_time_s"], 6),
            "Num. frames valid": int(analysis["n_frames_valid"]),
        }
        for region in REGIONS:
            row[f"Mean CoF {region} (%)"] = round(
                analysis["mean_pct"][region] * 100, 6
            )
        for region in REGIONS:
            row[f"Frames CoP {region} (%)"] = round(
                analysis["pct_cop"][region] * 100, 6
            )
        detail_rows.append(row)

    _write_csv(detail_path, DETAIL_FIELDS, detail_rows)

    summary_rows = _build_summary(detail_rows)
    _write_csv(summary_path, DETAIL_FIELDS, summary_rows)

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
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(headers), delimiter=';')
        writer.writeheader()
        for row in rows:
            # Formatear valores numéricos al estilo europeo
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, (int, float)) and key not in ["Participant id", "iPass", "FootstepID", "Num. frames valid"]:
                    formatted_row[key] = _format_number(value)
                else:
                    formatted_row[key] = value
            writer.writerow(formatted_row)


def _build_summary(
    detail_rows: Sequence[Dict[str, float | int | str]]
) -> List[Dict[str, float | int | str]]:
    def select_by_side(target: str) -> List[Dict[str, float | int | str]]:
        return [row for row in detail_rows if str(row["Side"]).lower() == target]

    summaries: List[Dict[str, float | int | str]] = []
    for label, selector in (
        ("Left", lambda: select_by_side("left")),
        ("Right", lambda: select_by_side("right")),
        ("Overall", lambda: list(detail_rows)),
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
    def mean(field: str) -> float:
        values = [float(row[field]) for row in rows]
        return float(np.mean(values)) if values else 0.0

    def sum_int(field: str) -> int:
        values = [int(row[field]) for row in rows]
        return int(sum(values))

    base = rows[0]
    aggregated: Dict[str, float | int | str] = {
        "Participant id": base["Participant id"],
        "Shoe type": base["Shoe type"],
        "Walk condition": base["Walk condition"],
        "iPass": "All",
        "FootstepID": "All",
        "Side": side_label,
        "Stance time (s)": round(mean("Stance time (s)"), 6),
        "Num. frames valid": sum_int("Num. frames valid"),
    }

    for region in REGIONS:
        aggregated[f"Mean CoF {region} (%)"] = round(
            mean(f"Mean CoF {region} (%)"), 6
        )
    for region in REGIONS:
        aggregated[f"Frames CoP {region} (%)"] = round(
            mean(f"Frames CoP {region} (%)"), 6
        )

    return aggregated
