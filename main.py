import os
import cv2
import numpy as np
import subprocess
import tempfile
import logging
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from skimage import filters, morphology, exposure
from scipy import ndimage
import pandas as pd
from PIL import Image
import io

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes
TESSERACT_PATH = "/opt/render/project/src/bin/tesseract"
TESSDATA_PATH = "/opt/render/project/src/tessdata"
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"}

app = FastAPI(
    title="OCR AIDA Pro", description="API avanzada para documentos académicos"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== UTILIDADES ====================
def validate_file(file: UploadFile):
    """Valida extensión y tamaño del archivo."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400, f"Formato no soportado: {ext}. Use {SUPPORTED_EXTENSIONS}"
        )
    # Nota: no podemos leer el tamaño aquí porque perderíamos el contenido, se hará después


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
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Trabajamos en RGB


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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
    if result.returncode != 0:
        logger.error(f"Tesseract error: {result.stderr}")
        raise RuntimeError(f"Error en OCR: {result.stderr}")
    return result.stdout.strip()


# ==================== PREPROCESAMIENTO ====================
def correct_skew(image: np.ndarray) -> np.ndarray:
    """Corrige la inclinación del documento. Acepta RGB (3 canales) o gris (1 canal)."""
    # Si es RGB, convertir a gris; si ya es gris, usarla directamente
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
            (h, w) = image.shape[:2]  # usa la imagen original (puede ser RGB o gris)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
    return image


def remove_noise(image: np.ndarray, method="nlmeans") -> np.ndarray:
    """Elimina ruido usando diferentes métodos. Acepta RGB o gris."""
    # Si es RGB, convertir a gris; si ya es gris, usarla directamente
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    logger.info(f"remove_noise - shape: {image.shape}, canales: {len(image.shape)}")

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
        # Implementación simple de Sauvola (requiere skimage)
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
    # 1. Corregir inclinación
    img = correct_skew(image)
    # 2. Eliminar sombras
    img = remove_shadows(img)
    # 3. Reducir ruido
    img = remove_noise(img, method="nlmeans")
    # 4. Binarizar
    img = binarize(img, method="sauvola" if img.mean() < 200 else "adaptive")
    return img


# ==================== DETECCIÓN DE TABLAS ====================
def detect_tables(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detecta regiones de tablas basadas en líneas horizontales y verticales.
    Retorna lista de dict con bounding boxes (x, y, w, h) y tipo.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Umbral para resaltar líneas
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detectar líneas horizontales
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    # Detectar líneas verticales
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    # Combinar líneas
    table_structure = cv2.add(horizontal_lines, vertical_lines)

    # Encontrar contornos de posibles tablas
    contours, _ = cv2.findContours(
        table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtrar por área mínima y relación de aspecto (típico de tablas)
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
    """
    Segmenta una región de tabla en celdas individuales.
    Retorna lista de imágenes de celdas.
    """
    x, y, w, h = bbox
    roi = image[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detectar líneas divisorias (horizontal y vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
    horizontal_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    vertical_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    # Encontrar intersecciones (esquinas de celdas)
    intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
    # Encontrar contornos de celdas (regiones entre líneas)
    # Un método más robusto es usar proyecciones, pero simplificamos:
    # Usamos connected components para encontrar bloques de texto que no sean líneas
    # En su lugar, haremos un enfoque más simple: asumimos que las líneas definen una cuadrícula.
    # Detectamos líneas horizontales y verticales, luego obtenemos sus posiciones.
    # (Implementación abreviada por espacio; se puede ampliar)

    # Por ahora retornamos la ROI completa para OCR general
    return [roi]


# ==================== SEGMENTACIÓN POR REGIONES ====================
def segment_regions(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Segmenta la imagen en regiones (texto, tablas, imágenes) usando análisis de componentes conectados.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilatar para unir componentes cercanos (párrafos)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 500:  # ignorar ruido
            continue
        # Clasificar por densidad de píxeles y aspecto
        roi = thresh[y : y + h, x : x + w]
        text_density = np.sum(roi) / (w * h * 255)  # proporción de píxeles de texto
        aspect = w / h
        if aspect > 3 and h < 50:  # posible línea separadora
            region_type = "line"
        elif text_density > 0.2:
            region_type = "text"
        else:
            region_type = "image"
        regions.append(
            {"bbox": (x, y, w, h), "type": region_type, "confidence": text_density}
        )
    return regions


# ==================== OCR POR REGIONES ====================
def ocr_region(image: np.ndarray, region: dict, lang: str) -> str:
    """Ejecuta OCR en una región específica con PSM adecuado según el tipo."""
    x, y, w, h = region["bbox"]
    roi = image[y : y + h, x : x + w]

    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name

    try:
        if region["type"] == "table":
            # Para tablas, usamos PSM 6 (uniform block) o 11 (sparse text)
            text = run_tesseract(tmp_path, lang, psm=6)
        elif region["type"] == "text":
            # Texto normal, PSM 3 (auto) o 4 (single column)
            text = run_tesseract(tmp_path, lang, psm=4)
        else:
            # Imágenes o líneas, PSM 8 (single word) o 7 (single line)
            text = run_tesseract(tmp_path, lang, psm=7)
    finally:
        os.unlink(tmp_path)
    return text


# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    """Health check con información de Tesseract."""
    return {
        "message": "API OCR AIDA Pro funcionando",
        "status": "ok",
        "tesseract": {
            "path": TESSERACT_PATH,
            "exists": os.path.exists(TESSERACT_PATH),
            "executable": os.access(TESSERACT_PATH, os.X_OK),
        },
        "tessdata": {
            "path": TESSDATA_PATH,
            "exists": os.path.exists(TESSDATA_PATH),
            "languages": [f.stem for f in Path(TESSDATA_PATH).glob("*.traineddata")],
        },
    }


@app.post("/ocr/basico")
async def ocr_basico(file: UploadFile = File(...), lang: str = Form("spa")):
    """Endpoint simple sin preprocesamiento (compatibilidad)."""
    validate_file(file)
    img = await read_image(file)

    # Guardar temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name

    try:
        text = run_tesseract(tmp_path, lang, psm=6)
    finally:
        os.unlink(tmp_path)

    return {
        "success": True,
        "filename": file.filename,
        "text": text,
        "metadata": {"language": lang},
    }


@app.post("/ocr/preprocesado")
async def ocr_con_preprocesamiento(
    file: UploadFile = File(...),
    lang: str = Form("spa"),
    correccion_skew: bool = Form(True),
    metodo_binarizacion: str = Form("sauvola"),  # otsu, adaptive, sauvola
):
    """OCR con pipeline de limpieza optimizado para documentos académicos."""
    validate_file(file)
    img = await read_image(file)

    # Aplicar preprocesamiento
    if correccion_skew:
        img = correct_skew(img)
    img = remove_shadows(img)
    img = remove_noise(img, method="nlmeans")
    img = binarize(img, method=metodo_binarizacion)

    # Guardar temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, img)  # ya está en gris o binario
        tmp_path = tmp.name

    try:
        text = run_tesseract(tmp_path, lang, psm=6)
    finally:
        os.unlink(tmp_path)

    return {
        "success": True,
        "filename": file.filename,
        "text": text,
        "metadata": {
            "language": lang,
            "skew_correction": correccion_skew,
            "binarization": metodo_binarizacion,
        },
    }


@app.post("/ocr/segmentado")
async def ocr_con_segmentacion(
    file: UploadFile = File(...),
    lang: str = Form("spa"),
    detectar_tablas: bool = Form(True),
):
    """
    Segmenta la imagen en regiones (texto, tablas, imágenes) y aplica OCR específico.
    Ideal para documentos complejos con múltiples elementos.
    """
    validate_file(file)
    img = await read_image(file)

    # Preprocesamiento ligero para mejorar segmentación
    processed = deskew_and_clean(img.copy())

    # Detectar regiones
    regions = segment_regions(processed)

    # Si se solicitó detección de tablas, añadirlas (pueden solaparse con regiones de texto)
    if detectar_tablas:
        tables = detect_tables(processed)
        # Fusionar regiones (evitar duplicados)
        for table in tables:
            # Verificar si alguna región de texto contiene esta tabla
            contained = False
            for reg in regions:
                rx, ry, rw, rh = reg["bbox"]
                tx, ty, tw, th = table["bbox"]
                if rx <= tx and ry <= ty and rx + rw >= tx + tw and ry + rh >= ty + th:
                    contained = True
                    break
            if not contained:
                regions.append(table)

    # Ordenar regiones de arriba a abajo, izquierda a derecha
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    # Aplicar OCR a cada región
    resultados = []
    for i, reg in enumerate(regions):
        try:
            texto = ocr_region(img, reg, lang)
            resultados.append(
                {"region": i, "tipo": reg["type"], "bbox": reg["bbox"], "texto": texto}
            )
        except Exception as e:
            logger.error(f"Error en región {i}: {e}")
            resultados.append(
                {"region": i, "tipo": reg["type"], "bbox": reg["bbox"], "error": str(e)}
            )

    # Texto completo (unión de todos los textos)
    texto_completo = "\n".join([r["texto"] for r in resultados if "texto" in r])

    return {
        "success": True,
        "filename": file.filename,
        "num_regiones": len(resultados),
        "texto_completo": texto_completo,
        "regiones": resultados,
        "metadata": {"language": lang, "detectar_tablas": detectar_tablas},
    }


@app.post("/ocr/tabla")
async def ocr_tabla(
    file: UploadFile = File(...),
    lang: str = Form("spa"),
    formato_salida: str = Form("json"),  # json, csv, dataframe
):
    """
    Especializado en extraer tablas: detecta la tabla, segmenta celdas y devuelve estructura.
    """
    validate_file(file)
    img = await read_image(file)

    # Preprocesar
    processed = deskew_and_clean(img)

    # Detectar tablas
    tables = detect_tables(processed)
    if not tables:
        return {"success": False, "error": "No se detectaron tablas en la imagen"}

    # Tomar la tabla más grande (asumimos que es la principal)
    main_table = max(tables, key=lambda t: t["bbox"][2] * t["bbox"][3])
    bbox = main_table["bbox"]

    # Extraer celdas (implementación básica, podría mejorarse)
    roi = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar líneas divisorias
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (roi.shape[1] // 10, 1)
    )
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi.shape[0] // 10))
    horizontal_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    vertical_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    # Encontrar coordenadas de líneas mediante proyección
    h_proj = np.sum(horizontal_lines, axis=1) // 255
    v_proj = np.sum(vertical_lines, axis=0) // 255

    # Umbral para detectar picos
    h_thresh = np.max(h_proj) * 0.2
    v_thresh = np.max(v_proj) * 0.2
    h_lines = np.where(h_proj > h_thresh)[0]
    v_lines = np.where(v_proj > v_thresh)[0]

    # Agrupar líneas cercanas (para evitar duplicados)
    def group_lines(lines, threshold=10):
        if len(lines) == 0:
            return []
        grouped = []
        current = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] - current[-1] <= threshold:
                current.append(lines[i])
            else:
                grouped.append(int(np.mean(current)))
                current = [lines[i]]
        grouped.append(int(np.mean(current)))
        return grouped

    h_pos = group_lines(h_lines)
    v_pos = group_lines(v_lines)

    # Crear cuadrícula de celdas
    data = []
    for i in range(len(h_pos) - 1):
        row = []
        for j in range(len(v_pos) - 1):
            x1, x2 = v_pos[j], v_pos[j + 1]
            y1, y2 = h_pos[i], h_pos[i + 1]
            cell_roi = roi[y1:y2, x1:x2]
            # Guardar celda temporalmente para OCR
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                cv2.imwrite(tmp.name, cv2.cvtColor(cell_roi, cv2.COLOR_RGB2BGR))
                tmp_path = tmp.name
            try:
                cell_text = run_tesseract(
                    tmp_path, lang, psm=7
                )  # PSM 7 para línea única
            except:
                cell_text = ""
            finally:
                os.unlink(tmp_path)
            row.append(cell_text.strip())
        data.append(row)

    # Crear DataFrame si se necesita
    if formato_salida == "csv":
        df = pd.DataFrame(data)
        csv_string = df.to_csv(index=False, header=False)
        return JSONResponse(content={"success": True, "csv": csv_string})
    elif formato_salida == "dataframe":
        return {"success": True, "data": data}
    else:  # json
        return {
            "success": True,
            "filename": file.filename,
            "tabla": data,
            "num_filas": len(data),
            "num_columnas": len(data[0]) if data else 0,
        }


# Funciones auxiliares que reciben numpy array en lugar de UploadFile
async def procesar_con_preprocesamiento(
    img: np.ndarray, lang: str, correccion_skew: bool, metodo_binarizacion: str
):
    # Aplicar preprocesamiento
    if correccion_skew:
        img = correct_skew(img)
    img = remove_shadows(img)
    img = remove_noise(img, method="nlmeans")
    img = binarize(img, method=metodo_binarizacion)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
    try:
        text = run_tesseract(tmp_path, lang, psm=6)
    finally:
        os.unlink(tmp_path)
    return {
        "success": True,
        "filename": "procesado",  # Podrías pasar el nombre original si lo guardas
        "text": text,
        "metadata": {
            "language": lang,
            "skew_correction": correccion_skew,
            "binarization": metodo_binarizacion,
        },
    }


async def procesar_con_segmentacion(img: np.ndarray, lang: str, detectar_tablas: bool):
    # Similar a ocr_con_segmentacion pero recibiendo img directamente
    processed = deskew_and_clean(img.copy())
    regions = segment_regions(processed)
    if detectar_tablas:
        tables = detect_tables(processed)
        for table in tables:
            contained = False
            for reg in regions:
                rx, ry, rw, rh = reg["bbox"]
                tx, ty, tw, th = table["bbox"]
                if rx <= tx and ry <= ty and rx + rw >= tx + tw and ry + rh >= ty + th:
                    contained = True
                    break
            if not contained:
                regions.append(table)
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    resultados = []
    for i, reg in enumerate(regions):
        try:
            texto = ocr_region(img, reg, lang)
            resultados.append(
                {"region": i, "tipo": reg["type"], "bbox": reg["bbox"], "texto": texto}
            )
        except Exception as e:
            resultados.append(
                {"region": i, "tipo": reg["type"], "bbox": reg["bbox"], "error": str(e)}
            )
    texto_completo = "\n".join([r["texto"] for r in resultados if "texto" in r])
    return {
        "success": True,
        "filename": "procesado",
        "num_regiones": len(resultados),
        "texto_completo": texto_completo,
        "regiones": resultados,
        "metadata": {"language": lang, "detectar_tablas": detectar_tablas},
    }


async def procesar_como_tabla(img: np.ndarray, lang: str):
    # Versión simplificada de ocr_tabla que recibe imagen
    processed = deskew_and_clean(img)
    tables = detect_tables(processed)
    if not tables:
        return {"success": False, "error": "No se detectaron tablas"}
    main_table = max(tables, key=lambda t: t["bbox"][2] * t["bbox"][3])
    bbox = main_table["bbox"]
    roi = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (roi.shape[1] // 10, 1)
    )
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi.shape[0] // 10))
    horizontal_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    vertical_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )
    h_proj = np.sum(horizontal_lines, axis=1) // 255
    v_proj = np.sum(vertical_lines, axis=0) // 255
    h_thresh = np.max(h_proj) * 0.2
    v_thresh = np.max(v_proj) * 0.2
    h_lines = np.where(h_proj > h_thresh)[0]
    v_lines = np.where(v_proj > v_thresh)[0]

    def group_lines(lines, threshold=10):
        if len(lines) == 0:
            return []
        grouped = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] - grouped[-1] <= threshold:
                grouped[-1] = (grouped[-1] + lines[i]) // 2  # promedio simple
            else:
                grouped.append(lines[i])
        return grouped

    h_pos = group_lines(h_lines)
    v_pos = group_lines(v_lines)
    data = []
    for i in range(len(h_pos) - 1):
        row = []
        for j in range(len(v_pos) - 1):
            x1, x2 = v_pos[j], v_pos[j + 1]
            y1, y2 = h_pos[i], h_pos[i + 1]
            cell_roi = roi[y1:y2, x1:x2]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                cv2.imwrite(tmp.name, cv2.cvtColor(cell_roi, cv2.COLOR_RGB2BGR))
                tmp_path = tmp.name
            try:
                cell_text = run_tesseract(tmp_path, lang, psm=7)
            except:
                cell_text = ""
            finally:
                os.unlink(tmp_path)
            row.append(cell_text.strip())
        data.append(row)
    return {
        "success": True,
        "filename": "procesado",
        "tabla": data,
        "num_filas": len(data),
        "num_columnas": len(data[0]) if data else 0,
    }


@app.post("/ocr/documento_completo")
async def ocr_documento_completo(
    file: UploadFile = File(...),
    lang: str = Form("spa"),
    optimizar_para: str = Form("texto"),
):
    try:
        validate_file(file)
        img = await read_image(file)

        # Análisis rápido
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )
        num_horizontal = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:
                    num_horizontal += 1

        if num_horizontal > 10 or optimizar_para == "tablas":
            return await procesar_como_tabla(img, lang)
        elif optimizar_para == "mixto":
            return await procesar_con_segmentacion(img, lang, detectar_tablas=True)
        else:
            return await procesar_con_preprocesamiento(
                img, lang, correccion_skew=True, metodo_binarizacion="sauvola"
            )
    except Exception as e:
        logger.error("Error en /ocr/documento_completo", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
