# ocr/engine.py
import subprocess
import os
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, List
import logging
import pytesseract
from pytesseract import Output

from config import TESSERACT_PATH, TESSDATA_PATH

logger = logging.getLogger(__name__)

# Configurar pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH


def run_tesseract(
    image_path: str,
    lang: str,
    psm: int = 6,
    oem: int = 1,
    timeout: int = 180,
    config: str = "",
) -> str:
    """Ejecuta Tesseract y retorna el texto."""
    cmd = [
        TESSERACT_PATH,
        image_path,
        "stdout",
        "-l",
        lang,
        "--psm",
        str(psm),
        "--oem",
        str(oem),
    ]
    if config:
        cmd += config.split()
    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = TESSDATA_PATH

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env
    )
    if result.returncode != 0:
        logger.error(f"Tesseract error: {result.stderr}")
        raise RuntimeError(f"Error en OCR: {result.stderr}")
    return result.stdout.strip()


def ocr_region(image: np.ndarray, region: dict, lang: str, timeout: int = 180) -> str:
    """Ejecuta OCR en una región específica con PSM adecuado según el tipo."""
    x, y, w, h = region["bbox"]
    roi = image[y : y + h, x : x + w]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name

    try:
        if region["type"] == "table":
            text = run_tesseract(tmp_path, lang, psm=6, timeout=timeout)
        elif region["type"] == "text":
            text = run_tesseract(tmp_path, lang, psm=4, timeout=timeout)
        else:
            text = run_tesseract(tmp_path, lang, psm=7, timeout=timeout)
    finally:
        os.unlink(tmp_path)
    return text


def get_text_data(
    image_path: str, lang: str, psm: int = 3, timeout: int = 180
) -> List[Dict]:
    """
    Ejecuta Tesseract y retorna lista de dicts con:
    - text: palabra
    - bbox: (x, y, w, h)
    - conf: confianza
    - line_num: número de línea (agrupado por Tesseract)
    - block_num: número de bloque
    - par_num: número de párrafo
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    # Configuración de Tesseract: preservar espacios, PSM
    config = f"--psm {psm} -c preserve_interword_spaces=1"

    try:
        data = pytesseract.image_to_data(
            img, lang=lang, config=config, output_type=Output.DICT, timeout=timeout
        )
    except RuntimeError as e:
        if "timed out" in str(e).lower():
            # Relanzar como subprocess.TimeoutExpired para consistencia
            raise subprocess.TimeoutExpired(cmd="tesseract", timeout=timeout)
        else:
            raise

    text_regions = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        conf = float(data["conf"][i])
        text = data["text"][i].strip()
        if conf > 30 and text:  # Umbral de confianza
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            text_regions.append(
                {
                    "text": text,
                    "bbox": (x, y, w, h),
                    "conf": conf,
                    "line": data["line_num"][i],
                    "block": data["block_num"][i],
                    "par": data["par_num"][i],
                }
            )
    return text_regions


def group_words_into_lines(text_regions: List[Dict]) -> List[Dict]:
    """
    Agrupa palabras por line_num para formar líneas de texto.

    Args:
        text_regions: Lista de diccionarios de palabras (salida de get_text_data).

    Returns:
        Lista de diccionarios, cada uno con:
            - text: texto completo de la línea
            - bbox: (x, y, w, h) que engloba todas las palabras de la línea
            - words: lista de las palabras originales (opcional)
    """
    lines = {}
    for word in text_regions:
        line_num = word["line"]
        if line_num not in lines:
            lines[line_num] = {
                "text": word["text"],
                "bbox": list(word["bbox"]),  # (x, y, w, h) como lista mutable
                "words": [word],
                "conf": word["conf"],
            }
        else:
            # Concatenar texto con espacio
            lines[line_num]["text"] += " " + word["text"]
            # Expandir bbox para cubrir toda la línea
            x, y, w, h = lines[line_num]["bbox"]
            x2, y2, w2, h2 = word["bbox"]
            new_x = min(x, x2)
            new_y = min(y, y2)
            new_w = max(x + w, x2 + w2) - new_x
            new_h = max(y + h, y2 + h2) - new_y
            lines[line_num]["bbox"] = [new_x, new_y, new_w, new_h]
            lines[line_num]["words"].append(word)
            lines[line_num]["conf"] += word["conf"]

    # Convertir a lista y asegurar que bbox sea tupla para consistencia
    result = []
    for line in lines.values():
        line["bbox"] = tuple(line["bbox"])
        line["conf"] = line["conf"] / len(line["words"])
        result.append(line)
    return result
