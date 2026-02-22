# preprocessing/checkbox.py
import cv2
import numpy as np
import re
from typing import List, Dict, Any, Tuple
import logging

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
                    "type": "square",
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
                        "type": "circle",
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
            from ocr.engine import run_tesseract

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
                        "type": "inciso",
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
