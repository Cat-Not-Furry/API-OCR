# preprocessing/compression.py
import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ImageCompressor:
    """
    Clase para comprimir imágenes optimizando para OCR.
    Estrategia: reducir dimensiones manteniendo calidad suficiente para lectura.
    """

    def __init__(self, target_size_mb: float = 2.0, max_dimension: int = 1500):
        """
        Args:
            target_size_mb: Tamaño objetivo en MB (Render-friendly)
            max_dimension: Dimensión máxima (ancho o alto) en píxeles
        """
        self.target_size_bytes = target_size_mb * 1024 * 1024
        self.max_dimension = max_dimension

    def compress_for_ocr(
        self, image: np.ndarray, original_filename: str = ""
    ) -> np.ndarray:
        """
        Pipeline completo de compresión para OCR.
        """
        original_shape = image.shape
        logger.info(
            f"Comprimiendo imagen {original_filename}: "
            f"{original_shape[1]}x{original_shape[0]}"
        )

        # 1. Redimensionar si excede dimensión máxima
        image = self._resize_if_needed(image)

        # 2. Comprimir vía calidad JPEG (iterativo)
        image = self._compress_by_quality(image)

        # Calcular tamaño aproximado
        approx_size = self._estimate_size(image)
        logger.info(
            f"Imagen comprimida: {image.shape[1]}x{image.shape[0]}, "
            f"tamaño aprox: {approx_size:.2f} MB"
        )

        return image

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= self.max_dimension:
            return image

        if h > w:
            scale = self.max_dimension / h
            new_h = self.max_dimension
            new_w = int(w * scale)
        else:
            scale = self.max_dimension / w
            new_w = self.max_dimension
            new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Redimensionado: {w}x{h} -> {new_w}x{new_h}")
        return resized

    def _compress_by_quality(
        self, image: np.ndarray, min_quality: int = 75
    ) -> np.ndarray:
        qualities = [95, 85, 75, 65, 55, 45, 35, 25, 15]
        best_result = image
        best_size = self._estimate_size(image)

        for quality in qualities:
            if quality < min_quality:
                break

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param
            )
            compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
            current_size = len(encoded) / (1024 * 1024)

            if current_size <= self.target_size_bytes / (1024 * 1024):
                logger.info(
                    f"Calidad {quality} alcanza objetivo: {current_size:.2f} MB"
                )
                return compressed

            if current_size < best_size:
                best_size = current_size
                best_result = compressed

        logger.info(f"Usando mejor calidad disponible: {best_size:.2f} MB")
        return best_result

    def _estimate_size(self, image: np.ndarray) -> float:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param
        )
        return len(encoded) / (1024 * 1024)


# Función simple para usar directamente en endpoints
async def compress_image(
    image: np.ndarray, max_size_mb: float = 2.0, max_dimension: int = 1500
) -> np.ndarray:
    compressor = ImageCompressor(
        target_size_mb=max_size_mb, max_dimension=max_dimension
    )
    return compressor.compress_for_ocr(image)
