# preprocessing/checkbox.py
import cv2
import numpy as np
import re
import tempfile
import os
from typing import List, Dict, Any, Tuple
import logging
from ocr.engine import run_tesseract

logger = logging.getLogger(__name__)


def detect_checkboxes(
    image: np.ndarray,
    min_area: int = 50,
    max_area: int = 5000,
    square_tolerance: float = 0.2,
    circle_tolerance: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Detecta checkboxes en una imagen.

    Args:
        image: Imagen en escala de grises o RGB.
        min_area: Área mínima del contorno.
        max_area: Área máxima del contorno.
        square_tolerance: Tolerancia para relación de aspecto (0.8-1.2).
        circle_tolerance: Tolerancia para circularidad (>0.7).

    Returns:
        Lista de diccionarios con:
            - bbox: (x, y, w, h)
            - type: "square", "circle", "inciso"
            - marked: bool (True si está marcado)
            - confidence: float (0-100)
            - text: texto dentro del checkbox (para incisos)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Binarización inversa para que los objetos sean blancos sobre fondo negro
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    checkboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0

        # ---- Detectar cuadrados [ ] ----
        if 0.8 - square_tolerance < aspect_ratio < 1.2 + square_tolerance:
            # Es cuadrado, verificar si está marcado
            roi = gray[y : y + h, x : x + w]
            marked, confidence = _is_checkbox_marked(roi)
            checkboxes.append(
                {
                    "bbox": (x, y, w, h),
                    "tipo": "square",
                    "marked": marked,
                    "confidence": confidence,
                    "text": None,
                }
            )
            continue

        # ---- Detectar círculos O ----
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7 - circle_tolerance:
                roi = gray[y : y + h, x : x + w]
                marked, confidence = _is_checkbox_marked(roi)
                checkboxes.append(
                    {
                        "bbox": (x, y, w, h),
                        "tipo": "circle",
                        "marked": marked,
                        "confidence": confidence,
                        "text": None,
                    }
                )
                continue

        # ---- Detectar incisos como (a), (A), etc. ----
        # Son pequeños, con relación de aspecto alargada (por el paréntesis)
        if aspect_ratio > 1.5 and area < 500:
            roi = gray[y : y + h, x : x + w]
            # Usar OCR rápido para extraer el texto del inciso
            # Nota: Necesitamos pytesseract o nuestro run_tesseract con PSM 8

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                cv2.imwrite(tmp.name, roi)
                tmp_path = tmp.name
            try:
                text = run_tesseract(tmp_path, lang="spa+eng", psm=8).strip()
            except Exception as e:
                logger.debug(f"Error OCR en inciso: {e}")
                text = ""
            finally:
                os.unlink(tmp_path)

            # Validar que sea algo como (a), (A), (1), etc.
            if re.match(r"^\([a-zA-Z0-9]\)$", text):
                checkboxes.append(
                    {
                        "bbox": (x, y, w, h),
                        "tipo": "inciso",
                        "marked": False,  # Los incisos no se marcan, son etiquetas
                        "confidence": 100.0,
                        "text": text,
                    }
                )

    logger.info(f"Detectados {len(checkboxes)} checkboxes")
    return checkboxes


def _is_checkbox_marked(roi: np.ndarray, threshold: float = 0.1) -> Tuple[bool, float]:
    """
    Determina si un checkbox (cuadrado o círculo) está marcado.
    Analiza la proporción de píxeles oscuros en el interior.

    Args:
        roi: Región de interés (gris).
        threshold: Proporción mínima de píxeles oscuros para considerar marcado.

    Returns:
        (marked, confidence) donde confidence es el porcentaje de píxeles oscuros.
    """
    # Asegurar que la ROI tiene tamaño suficiente
    if roi.size == 0:
        return False, 0.0

    # Binarizar la ROI (inversa para que la marca sea blanca)
    _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calcular proporción de píxeles blancos (marca)
    white_pixels = np.sum(roi_thresh == 255)
    total_pixels = roi.size
    proportion = white_pixels / total_pixels if total_pixels > 0 else 0

    # Si hay una marca, la proporción suele ser > 0.1 (10%)
    marked = proportion > threshold
    confidence = proportion * 100  # Convertir a porcentaje
    return marked, confidence


def associate_checkboxes_with_text(
    checkboxes: List[Dict[str, Any]],
    text_regions: List[Dict[str, Any]],
    max_distance: int = 100,
) -> List[Dict[str, Any]]:
    """
    Asocia cada checkbox con la región de texto más cercana (la pregunta).

    Args:
        checkboxes: Lista de checkboxes (con bbox).
        text_regions: Lista de regiones de texto (con bbox y texto).
        max_distance: Distancia vertical máxima permitida (píxeles).

    Returns:
        Lista de checkboxes con campo 'associated_text' y 'distance'.
    """
    for cb in checkboxes:
        cb_center_y = cb["bbox"][1] + cb["bbox"][3] // 2
        best_match = None
        best_dist = float("inf")

        for tr in text_regions:
            tr_center_y = tr["bbox"][1] + tr["bbox"][3] // 2
            # La región debe estar arriba del checkbox (la pregunta suele estar antes)
            if tr_center_y < cb_center_y:
                dist = cb_center_y - tr_center_y
                if dist < max_distance and dist < best_dist:
                    best_dist = dist
                    best_match = tr.get("text", "")

        cb["associated_text"] = best_match if best_match else ""
        cb["distance_to_text"] = best_dist if best_dist != float("inf") else -1

    return checkboxes


def associate_checkboxes_with_text_advanced(
    checkboxes: List[Dict],
    text_lines: List[Dict],
    max_horizontal_distance: int = 150,
    max_vertical_distance: int = 50,
    consider_right: bool = True,
) -> List[Dict]:
    """
    Asocia cada checkbox con la línea de texto más cercana.
    Prioriza: misma línea (horizontal) > arriba cercana > izquierda cercana.
    Si consider_right=True, también busca texto a la derecha cuando no hay a la izquierda.

    Args:
        checkboxes: Lista de checkboxes (con bbox y tipo).
        text_lines: Lista de líneas de texto (de group_words_into_lines).
        max_horizontal_distance: Distancia horizontal máxima permitida.
        max_vertical_distance: Distancia vertical máxima para considerar misma línea.
        consider_right: Si debe buscar texto a la derecha del checkbox.

    Returns:
        Lista de checkboxes con campos:
            - associated_text: texto asociado
            - association_confidence: confianza (0-100)
            - association_side: "left", "right", "above", "below", o None
    """
    for cb in checkboxes:
        cb_x, cb_y, cb_w, cb_h = cb["bbox"]
        cb_center_x = cb_x + cb_w // 2
        cb_center_y = cb_y + cb_h // 2

        best_match = None
        best_score = float("inf")
        best_side = None

        for line in text_lines:
            lx, ly, lw, lh = line["bbox"]
            l_center_x = lx + lw // 2
            l_center_y = ly + lh // 2

            # Calcular distancias
            horizontal_dist = abs(cb_center_x - l_center_x)
            vertical_dist = abs(cb_center_y - l_center_y)

            # Caso 1: Misma línea (verticalmente cerca)
            if vertical_dist < max_vertical_distance:
                # Texto a la izquierda
                if lx + lw < cb_x:
                    score = horizontal_dist  # distancia horizontal
                    if score < best_score:
                        best_score = score
                        best_match = line["text"]
                        best_side = "left"
                # Texto a la derecha (si está permitido)
                if consider_right and cb_x + cb_w < lx:
                    score = horizontal_dist
                    if score < best_score:
                        best_score = score
                        best_match = line["text"]
                        best_side = "right"

            # Caso 2: Texto arriba
            elif ly + lh < cb_y and vertical_dist < max_vertical_distance * 2:
                # Ponderar: vertical más importante que horizontal
                score = vertical_dist + horizontal_dist * 0.5
                if score < best_score:
                    best_score = score
                    best_match = line["text"]
                    best_side = "above"

            # Caso 3: Texto abajo (si no hay otra opción, a veces la pregunta puede estar después)
            elif cb_y + cb_h < ly and vertical_dist < max_vertical_distance * 2:
                score = vertical_dist + horizontal_dist * 0.5
                if score < best_score:
                    best_score = score
                    best_match = line["text"]
                    best_side = "below"

        cb["associated_text"] = best_match if best_match else ""
        # Convertir score a confianza (inversamente proporcional, máximo 100)
        if best_match:
            # Normalizar: score máximo esperado ~200, confianza = max(0, 100 - score/2)
            cb["association_confidence"] = max(0, min(100, 100 - best_score / 2))
        else:
            cb["association_confidence"] = 0
        cb["association_side"] = best_side

    return checkboxes


def group_checkboxes_by_proximity(
    checkboxes: List[Dict], vertical_threshold: int = 30
) -> List[List[Dict]]:
    """
    Agrupa checkboxes que están cerca verticalmente (probablemente opciones de una misma pregunta).
    Útil para detectar radio buttons o grupos de opciones.

    Args:
        checkboxes: Lista de checkboxes.
        vertical_threshold: Distancia vertical máxima para considerar mismo grupo.

    Returns:
        Lista de grupos, cada grupo es una lista de checkboxes.
    """
    if not checkboxes:
        return []

    # Ordenar por coordenada Y
    sorted_cb = sorted(checkboxes, key=lambda cb: cb["bbox"][1])
    groups = []
    current_group = [sorted_cb[0]]

    for cb in sorted_cb[1:]:
        last_cb = current_group[-1]
        last_y = last_cb["bbox"][1] + last_cb["bbox"][3]
        current_y = cb["bbox"][1]
        if current_y - last_y < vertical_threshold:
            current_group.append(cb)
        else:
            groups.append(current_group)
            current_group = [cb]
    groups.append(current_group)
    return groups


def ensure_single_marked_per_group(
    checkboxes: List[Dict], groups: List[List[Dict]]
) -> List[Dict]:
    """
    Para grupos de checkboxes que actúan como radio buttons (solo uno debe estar marcado),
    ajusta las marcas si hay múltiples marcados, quedándose con el de mayor confianza.
    """
    for group in groups:
        marked = [cb for cb in group if cb.get("marked", False)]
        if len(marked) > 1:
            # Elegir el de mayor confianza
            best = max(marked, key=lambda cb: cb.get("confidence", 0))
            for cb in group:
                if cb is not best:
                    cb["marked"] = False
                    cb["confidence"] = 100 - cb.get(
                        "confidence", 0
                    )  # opcional: ajustar
    return checkboxes
