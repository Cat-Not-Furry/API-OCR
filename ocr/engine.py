# ocr/engine.py
import subprocess
import os
import tempfile
import cv2
import numpy as np
from typing import Dict, Any
import logging
from config import TESSERACT_PATH, TESSDATA_PATH

logger = logging.getLogger(__name__)


def run_tesseract(
    image_path: str, lang: str, psm: int = 6, oem: int = 3, config: str = ""
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    if result.returncode != 0:
        logger.error(f"Tesseract error: {result.stderr}")
        raise RuntimeError(f"Error en OCR: {result.stderr}")
    return result.stdout.strip()


def ocr_region(image: np.ndarray, region: dict, lang: str) -> str:
    """Ejecuta OCR en una región específica con PSM adecuado según el tipo."""
    x, y, w, h = region["bbox"]
    roi = image[y : y + h, x : x + w]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name

    try:
        if region["type"] == "table":
            text = run_tesseract(tmp_path, lang, psm=6)
        elif region["type"] == "text":
            text = run_tesseract(tmp_path, lang, psm=4)
        else:
            text = run_tesseract(tmp_path, lang, psm=7)
    finally:
        os.unlink(tmp_path)
    return text
