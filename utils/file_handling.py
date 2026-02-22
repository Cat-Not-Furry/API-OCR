# utils/file_handling.py
import cv2
import numpy as np
from fastapi import UploadFile, HTTPException
from pathlib import Path
import logging
from config import MAX_FILE_SIZE, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def validate_file(file: UploadFile):
    """Valida extensión y tamaño del archivo."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400, f"Formato no soportado: {ext}. Use {SUPPORTED_EXTENSIONS}"
        )


async def read_image(file: UploadFile) -> np.ndarray:
    """Lee un UploadFile y lo convierte en array numpy (RGB)."""
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            400, f"Archivo muy grande (máx {MAX_FILE_SIZE // 1024 // 1024} MB)"
        )
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "No se pudo decodificar la imagen")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
