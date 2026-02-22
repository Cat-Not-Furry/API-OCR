# main.py
import os
import cv2
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import tempfile
from pathlib import Path

# Importaciones de nuestros módulos
from config import TESSERACT_PATH, TESSDATA_PATH, DEFAULT_LANG
from utils.file_handling import validate_file, read_image
from utils.logging_config import setup_logging
from preprocessing.enhance import (
    correct_skew,
    remove_shadows,
    remove_noise,
    binarize,
    deskew_and_clean,
)
from preprocessing.detection import detect_tables, segment_regions
from ocr.engine import run_tesseract, ocr_region
from ocr.postprocess import clean_text
from integration.infinityfree import InfinityFreeClient

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# Inicializar cliente de InfinityFree (cuando esté listo)
# infinity_client = InfinityFreeClient("https://tusitio.infinityfreeapp.com")

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


# ==================== FUNCIONES AUXILIARES ====================
async def procesar_con_preprocesamiento(
    img: np.ndarray, lang: str, correccion_skew: bool, metodo_binarizacion: str
):
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
        text = clean_text(text)
    finally:
        os.unlink(tmp_path)
    return {"text": text}


async def procesar_con_segmentacion(img: np.ndarray, lang: str, detectar_tablas: bool):
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
            logger.error(f"Error en región {i}: {e}")
            resultados.append(
                {"region": i, "tipo": reg["type"], "bbox": reg["bbox"], "error": str(e)}
            )

    texto_completo = "\n".join([r["texto"] for r in resultados if "texto" in r])
    return {
        "num_regiones": len(resultados),
        "texto_completo": texto_completo,
        "regiones": resultados,
    }


async def procesar_como_tabla(img: np.ndarray, lang: str):
    processed = deskew_and_clean(img)
    tables = detect_tables(processed)

    if not tables:
        return {"error": "No se detectaron tablas"}

    main_table = max(tables, key=lambda t: t["bbox"][2] * t["bbox"][3])
    bbox = main_table["bbox"]
    roi = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detección básica de líneas (versión simplificada)
    # ... (código de detección de líneas)

    # Por ahora devolvemos la tabla como texto plano
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name
    try:
        text = run_tesseract(tmp_path, lang, psm=6)
    finally:
        os.unlink(tmp_path)

    return {"tabla_texto": text, "bbox": bbox}


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
async def ocr_basico(file: UploadFile = File(...), lang: str = Form(DEFAULT_LANG)):
    """Endpoint simple sin preprocesamiento."""
    validate_file(file)
    img = await read_image(file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name

    try:
        text = run_tesseract(tmp_path, lang, psm=6)
        text = clean_text(text)
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
    lang: str = Form(DEFAULT_LANG),
    correccion_skew: bool = Form(True),
    metodo_binarizacion: str = Form("sauvola"),
):
    """OCR con pipeline de limpieza optimizado."""
    validate_file(file)
    img = await read_image(file)

    resultado = await procesar_con_preprocesamiento(
        img, lang, correccion_skew, metodo_binarizacion
    )

    return {
        "success": True,
        "filename": file.filename,
        "text": resultado["text"],
        "metadata": {
            "language": lang,
            "skew_correction": correccion_skew,
            "binarization": metodo_binarizacion,
        },
    }


@app.post("/ocr/segmentado")
async def ocr_con_segmentacion(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    detectar_tablas: bool = Form(True),
):
    """Segmenta la imagen y aplica OCR específico por región."""
    validate_file(file)
    img = await read_image(file)

    resultado = await procesar_con_segmentacion(img, lang, detectar_tablas)

    return {
        "success": True,
        "filename": file.filename,
        "num_regiones": resultado["num_regiones"],
        "texto_completo": resultado["texto_completo"],
        "regiones": resultado["regiones"],
        "metadata": {"language": lang, "detectar_tablas": detectar_tablas},
    }


@app.post("/ocr/documento_completo")
async def ocr_documento_completo(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    optimizar_para: str = Form("texto"),
):
    """Endpoint inteligente que elige el mejor método según el documento."""
    try:
        validate_file(file)
        img = await read_image(file)

        # Análisis rápido del documento
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
            resultado = await procesar_como_tabla(img, lang)
            if "error" in resultado:
                # Si falla como tabla, intentar como segmentado
                resultado = await procesar_con_segmentacion(
                    img, lang, detectar_tablas=True
                )
        elif optimizar_para == "mixto":
            resultado = await procesar_con_segmentacion(img, lang, detectar_tablas=True)
        else:
            resultado = await procesar_con_preprocesamiento(
                img, lang, correccion_skew=True, metodo_binarizacion="sauvola"
            )

        return {
            "success": True,
            "filename": file.filename,
            **resultado,
            "metadata": {
                "language": lang,
                "optimizacion": optimizar_para,
                "lineas_detectadas": num_horizontal,
            },
        }

    except Exception as e:
        logger.error("Error en /ocr/documento_completo", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
