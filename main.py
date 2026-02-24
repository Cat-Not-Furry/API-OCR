# ocr/association.py
from typing import List, Dict, Any
import logging
import tempfile
import os
import cv2
import numpy as np

from preprocessing.checkbox import (
    associate_checkboxes_with_text_advanced,
    group_checkboxes_by_proximity,
    ensure_single_marked_per_group,
)
from ocr.engine import run_tesseract

logger = logging.getLogger(__name__)


def group_words_into_lines(text_regions: List[Dict]) -> List[Dict]:
    """
    Agrupa palabras por line_num para formar líneas de texto completas.
    Cada línea incluye el texto concatenado y un bbox que engloba todas las palabras.

    Args:
        text_regions: Lista de palabras con bbox y line_num (de get_text_data).

    Returns:
        Lista de diccionarios con:
            - text: texto completo de la línea
            - bbox: (x, y, w, h) que cubre toda la línea
            - words: lista de palabras originales
    """
    lines = {}
    for word in text_regions:
        line_num = word["line"]
        if line_num not in lines:
            lines[line_num] = {
                "text": word["text"],
                "bbox": list(word["bbox"]),
                "words": [word],
            }
        else:
            lines[line_num]["text"] += " " + word["text"]
            # Expandir bbox para cubrir toda la línea
            x, y, w, h = lines[line_num]["bbox"]
            x2, y2, w2, h2 = word["bbox"]
            new_x = min(x, x2)
            new_y = min(y, y2)
            new_w = max(x + w, x2 + w2) - new_x
            new_h = max(y + h, y2 + h2) - new_y
            lines[line_num]["bbox"] = (new_x, new_y, new_w, new_h)
            lines[line_num]["words"].append(word)

    return list(lines.values())


def build_question_answer_pairs(
    checkboxes: List[Dict],
    text_lines: List[Dict],
    image_path: str = None,
    lang: str = "spa",
) -> List[Dict]:
    """
    Construye una lista de pares pregunta-respuesta a partir de los checkboxes
    y las líneas de texto detectadas.

    Args:
        checkboxes: Lista de checkboxes (con bbox, marked, tipo, etc.).
        text_lines: Lista de líneas de texto (con text y bbox).
        image_path: Ruta a la imagen (opcional, para OCR adicional si es necesario).
        lang: Idioma para OCR adicional.

    Returns:
        Lista de diccionarios con:
            - question: texto de la línea asociada
            - answer: "X" si marcado, "" si no
            - checkbox_bbox: bbox del checkbox
            - confidence: confianza de la asociación
            - side: lado de la asociación
    """
    # 1. Asociar checkboxes con texto usando heurística avanzada
    checkboxes_with_text = associate_checkboxes_with_text_advanced(
        checkboxes, text_lines, consider_right=True
    )

    # 2. Agrupar checkboxes por proximidad (para radio buttons)
    groups = group_checkboxes_by_proximity(checkboxes_with_text)

    # 3. Asegurar que en cada grupo solo haya un marcado (si es necesario)
    checkboxes_fixed = ensure_single_marked_per_group(checkboxes_with_text, groups)

    # 4. Construir pares pregunta-respuesta
    qa_pairs = []
    for cb in checkboxes_fixed:
        if cb.get("associated_text"):
            qa_pairs.append(
                {
                    "question": cb["associated_text"],
                    "answer": "X" if cb.get("marked") else "",
                    "checkbox_bbox": cb["bbox"],
                    "confidence": cb.get("association_confidence", 0),
                    "side": cb.get("association_side"),
                }
            )
        else:
            # Si no se pudo asociar, intentar extraer texto justo a la derecha (pregunta en imagen)
            if image_path and os.path.exists(image_path):
                # Opcional: recortar región a la derecha y hacer OCR rápido
                # (implementación simplificada, se puede ampliar)
                logger.debug(f"Checkbox sin texto asociado en {cb['bbox']}")
                qa_pairs.append(
                    {
                        "question": "",
                        "answer": "X" if cb.get("marked") else "",
                        "checkbox_bbox": cb["bbox"],
                        "confidence": 0,
                        "side": None,
                    }
                )
            else:
                qa_pairs.append(
                    {
                        "question": "",
                        "answer": "X" if cb.get("marked") else "",
                        "checkbox_bbox": cb["bbox"],
                        "confidence": 0,
                        "side": None,
                    }
                )

    return qa_pairs
