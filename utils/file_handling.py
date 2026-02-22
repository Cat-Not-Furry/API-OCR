# utils/file_handling.py (añadir al final o donde corresponda)
import cv2
import numpy as np
from fastapi import UploadFile, HTTPException
from pathlib import Path
import logging
from config import MAX_FILE_SIZE, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def resize_if_needed(image: np.ndarray, max_width: int = 2000) -> np.ndarray:
    """
    Redimensiona la imagen si el ancho supera max_width, manteniendo la relación de aspecto.
    Usa interpolación INTER_AREA para reducir calidad de forma aceptable.
    """
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_size = (max_width, int(h * scale))
        # INTER_AREA es mejor para reducir
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def validate_file(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400, f"Formato no soportado: {ext}. Use {SUPPORTED_EXTENSIONS}"
        )


async def read_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            400, f"Archivo muy grande (máx {MAX_FILE_SIZE // 1024 // 1024} MB)"
        )
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "No se pudo decodificar la imagen")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Redimensionar antes de devolver
    rgb = resize_if_needed(rgb, max_width=2000)
    return rgb
