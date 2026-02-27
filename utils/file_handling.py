# utils/file_handling.py
import cv2
import numpy as np
from fastapi import UploadFile, HTTPException
from pathlib import Path
import logging
from config import MAX_FILE_SIZE, SUPPORTED_EXTENSIONS
from preprocessing.compression import compress_image

logger = logging.getLogger(__name__)


def validate_file(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400, f"Formato no soportado: {ext}. Use {SUPPORTED_EXTENSIONS}"
        )


async def read_image(
    file: UploadFile, compress: bool = True, max_size_mb: float = 1.5
) -> tuple[np.ndarray, int]:
    """
    Lee un UploadFile y lo convierte en array numpy (RGB).
    Retorna (imagen_rgb, tamaño_original_en_bytes).
    Si compress=True y la imagen supera 1 MB, la comprime automáticamente.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "El archivo está vacío o no se pudo leer.")

    original_size = len(contents)
    original_size_mb = original_size / (1024 * 1024)
    logger.info(f"Tamaño original: {original_size_mb:.2f} MB")

    # Decodificar
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "No se pudo decodificar la imagen")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    max_dim = 2000  # Ajusta según necesidad
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Redimensionado a {new_w}x{new_h} píxeles")

    # Comprimir si se solicita y el archivo es grande
    if compress and original_size_mb > 1.0:
        logger.info("Aplicando compresión para reducir tamaño...")
        img_rgb = await compress_image(img_rgb, max_size_mb=max_size_mb)

        # Calcular nuevo tamaño aproximado (solo para log)
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        new_size_mb = len(buffer) / (1024 * 1024)
        logger.info(f"Tamaño después de compresión: {new_size_mb:.2f} MB")

    return img_rgb, original_size
