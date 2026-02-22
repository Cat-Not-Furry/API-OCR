# preprocessing/detection.py
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_tables(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detecta regiones de tablas."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    table_structure = cv2.add(horizontal_lines, vertical_lines)

    contours, _ = cv2.findContours(
        table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 5000 and w > 100 and h > 100 and (w / h) < 10:
            tables.append(
                {
                    "bbox": (x, y, w, h),
                    "type": "table",
                    "confidence": float(area) / (image.shape[0] * image.shape[1]),
                }
            )
    return tables


def extract_table_cells(image: np.ndarray, bbox: tuple) -> List[np.ndarray]:
    """Extrae celdas de una tabla (versión simplificada)."""
    x, y, w, h = bbox
    roi = image[y : y + h, x : x + w]
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return [roi]  # Por ahora retornamos ROI completa


def segment_regions(image: np.ndarray) -> List[Dict[str, Any]]:
    """Segmenta la imagen en regiones (texto, líneas, imágenes)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 500:
            continue
        roi = thresh[y : y + h, x : x + w]
        text_density = np.sum(roi) / (w * h * 255)
        aspect = w / h
        if aspect > 3 and h < 50:
            region_type = "line"
        elif text_density > 0.2:
            region_type = "text"
        else:
            region_type = "image"
        regions.append(
            {"bbox": (x, y, w, h), "type": region_type, "confidence": text_density}
        )
    return regions
