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


def resize_for_ocr(image: np.ndarray, target_width=2000) -> np.ndarray:
    h, w = image.shape[:2]
    if w < target_width:
        scale = target_width / w
        new_size = (target_width, int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return image


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


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Aplica CLAHE para mejorar contraste."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced


def deskew_and_clean(image: np.ndarray) -> np.ndarray:
    # 0. Detectar y corregir perspectiva
    img = detect_document_contour(image)
    # 1. Corregir inclinación residual
    img = correct_skew(img)

    img = remove_noise(img, method="nlmeans")
    img = binarize(img, method="sauvola" if img.mean() < 200 else "adaptive")
    return img


def detect_document_contour(image: np.ndarray) -> np.ndarray:
    """Detecta el contorno más grande (probable documento) y lo recorta en perspectiva."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Tomar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    # Si tiene 4 puntos, asumimos que es el documento (puede ser más si es irregular)
    if len(approx) == 4:
        # Ordenar puntos: superior-izquierdo, superior-derecho, inferior-derecho, inferior-izquierdo
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # superior-izquierdo
        rect[2] = pts[np.argmax(s)]  # inferior-derecho
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # superior-derecho
        rect[3] = pts[np.argmax(diff)]  # inferior-izquierdo

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    return image


def try_multiple_preprocessings(img: np.ndarray, lang: str) -> str:
    pipelines = [
        lambda x: deskew_and_clean(x),
        lambda x: apply_clahe(deskew_and_clean(x)),
        lambda x: binarize(remove_shadows(correct_skew(x)), method="otsu"),
        # ... añade más combinaciones
    ]
    best_text = ""
    max_words = 0
    for pipe in pipelines:
        processed = pipe(img.copy())
        # Guardar temp y ejecutar Tesseract rápido con PSM 3 (auto)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            cv2.imwrite(tmp.name, processed)
            text = run_tesseract(tmp.name, lang, psm=3)
            words = len(text.split())
            if words > max_words:
                max_words = words
                best_text = text
    return best_text


def correct_spelling(text: str, lang="es") -> str:
    spell = SpellChecker(language=lang)
    words = text.split()
    corrected = [spell.correction(w) if spell.correction(w) else w for w in words]
    return " ".join(corrected)
