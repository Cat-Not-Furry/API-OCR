# preprocessing/enhance.py
import cv2
import numpy as np
from skimage import filters
import logging

logger = logging.getLogger(__name__)


def correct_skew(image: np.ndarray) -> np.ndarray:
    """Corrige la inclinación del documento."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.rad2deg(theta) - 90
            angles.append(angle)
        median_angle = np.median(angles)
        if abs(median_angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
    return image


def remove_noise(image: np.ndarray, method="nlmeans") -> np.ndarray:
    """Elimina ruido usando diferentes métodos."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    logger.debug(f"remove_noise - shape: {image.shape}")

    if method == "nlmeans":
        return cv2.fastNlMeansDenoising(
            gray, h=30, templateWindowSize=7, searchWindowSize=21
        )
    elif method == "gaussian":
        return cv2.GaussianBlur(gray, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(gray, 5)
    elif method == "bilateral":
        return cv2.bilateralFilter(gray, 9, 75, 75)
    else:
        return gray


def binarize(image: np.ndarray, method="adaptive") -> np.ndarray:
    """Binarización avanzada: Otsu, Adaptive o Sauvola."""
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
    elif method == "sauvola":
        from skimage.filters import threshold_sauvola

        window_size = 25
        thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
        thresh = (gray > thresh_sauvola).astype(np.uint8) * 255
    else:
        thresh = gray
    return thresh


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """Elimina sombras mediante corrección de iluminación."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    blurred = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, blurred)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm


def deskew_and_clean(image: np.ndarray) -> np.ndarray:
    """Pipeline completo de limpieza para documentos."""
    img = correct_skew(image)
    img = remove_shadows(img)
    img = remove_noise(img, method="nlmeans")
    img = binarize(img, method="sauvola" if img.mean() < 200 else "adaptive")
    return img
