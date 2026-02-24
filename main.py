# main.py
import os
import cv2
import logging
import subprocess  # <-- AÑADIDO para capturar TimeoutExpired
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import tempfile
from pathlib import Path
from ocr.engine import get_text_data

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
from ocr.postprocess import clean_text, estructurar_texto_ocr
from preprocessing.detection import detect_text_fields
from preprocessing.checkbox import detect_checkboxes, associate_checkboxes_with_text
from integration.infinityfree import InfinityFreeClient
from config import INFINITYFREE_URL

infinity_client = InfinityFreeClient(INFINITYFREE_URL)

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

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
        # <-- MODIFICADO: capturar timeout
        try:
            text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Tiempo de espera agotado durante el preprocesamiento. La imagen es demasiado compleja.",
            )
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
            # <-- MODIFICADO: capturar timeout en OCR de región
            try:
                texto = ocr_region(img, reg, lang, timeout=120)
            except subprocess.TimeoutExpired:
                raise HTTPException(
                    status_code=504,
                    detail=f"Tiempo de espera agotado en región {i} (tipo: {reg['type']}).",
                )
            resultados.append(
                {"region": i, "tipo": reg["type"], "bbox": reg["bbox"], "texto": texto}
            )
        except Exception as e:
            logger.error(f"Error en región {i}: {e}")
            # Si ya es HTTPException, la relanzamos para que FastAPI la maneje
            if isinstance(e, HTTPException):
                raise
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name
    try:
        # <-- MODIFICADO: capturar timeout
        try:
            text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504, detail="Tiempo de espera agotado al procesar la tabla."
            )
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
async def ocr_basico(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    correct_spelling: bool = Form(False),
):
    validate_file(file)
    img = await read_image(file, compress=True, max_size_mb=2.0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name
    try:
        # <-- MODIFICADO: capturar timeout
        try:
            text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Tiempo de espera agotado. La imagen es demasiado compleja.",
            )
        text = clean_text(text)
        estructurado = estructurar_texto_ocr(
            text, corregir_ortografia_flag=correct_spelling
        )
    finally:
        os.unlink(tmp_path)
    return {
        "success": True,
        "filename": file.filename,
        "text": text,
        "texto_estructurado": estructurado,
        "metadata": {"language": lang, "correct_spelling": correct_spelling},
    }


@app.post("/ocr/segmentado")
async def ocr_con_segmentacion(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    detectar_tablas: bool = Form(True),
    correct_spelling: bool = Form(False),
):
    validate_file(file)
    img = await read_image(file, compress=True, max_size_mb=2.0)

    # procesar_con_segmentacion ya maneja timeout y lanza HTTPException
    resultado = await procesar_con_segmentacion(img, lang, detectar_tablas)

    texto_completo = resultado["texto_completo"]
    estructurado = estructurar_texto_ocr(
        texto_completo, corregir_ortografia_flag=correct_spelling
    )

    return {
        "success": True,
        "filename": file.filename,
        "num_regiones": resultado["num_regiones"],
        "texto_completo": texto_completo,
        "texto_estructurado": estructurado,
        "regiones": resultado["regiones"],
        "metadata": {
            "language": lang,
            "detectar_tablas": detectar_tablas,
            "correct_spelling": correct_spelling,
        },
    }


@app.post("/ocr/tabla")
async def ocr_tabla(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    formato_salida: str = Form("json"),
    correct_spelling: bool = Form(False),
):
    validate_file(file)
    img = await read_image(file, compress=True, max_size_mb=2.0)

    # procesar_como_tabla ya maneja timeout y lanza HTTPException
    resultado = await procesar_como_tabla(img, lang)
    if "error" in resultado:
        return {"success": False, "error": resultado["error"]}

    tabla_texto = resultado["tabla_texto"]
    estructurado = estructurar_texto_ocr(
        tabla_texto, corregir_ortografia_flag=correct_spelling
    )

    return {
        "success": True,
        "filename": file.filename,
        "tabla_texto": tabla_texto,
        "texto_estructurado": estructurado,
        "bbox": resultado["bbox"],
        "metadata": {
            "language": lang,
            "formato_salida": formato_salida,
            "correct_spelling": correct_spelling,
        },
    }


@app.post("/ocr/documento_completo")
async def ocr_documento_completo(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    optimizar_para: str = Form("texto"),
    correct_spelling: bool = Form(False),
):
    try:
        validate_file(file)
        img = await read_image(file, compress=True, max_size_mb=2.0)

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

        # Las funciones auxiliares ya manejan timeout
        if num_horizontal > 10 or optimizar_para == "tablas":
            resultado = await procesar_como_tabla(img, lang)
            if "error" in resultado:
                resultado = await procesar_con_segmentacion(
                    img, lang, detectar_tablas=True
                )
        elif optimizar_para == "mixto":
            resultado = await procesar_con_segmentacion(img, lang, detectar_tablas=True)
        else:
            resultado = await procesar_con_preprocesamiento(
                img, lang, correccion_skew=True, metodo_binarizacion="sauvola"
            )

        # Determinar qué tipo de texto se obtuvo
        if "text" in resultado:
            texto = resultado["text"]
        elif "texto_completo" in resultado:
            texto = resultado["texto_completo"]
        elif "tabla_texto" in resultado:
            texto = resultado["tabla_texto"]
        else:
            texto = ""

        estructurado = (
            estructurar_texto_ocr(texto, corregir_ortografia_flag=correct_spelling)
            if texto
            else {}
        )

        return {
            "success": True,
            "filename": file.filename,
            **resultado,
            "texto_estructurado": estructurado,
            "metadata": {
                "language": lang,
                "optimizacion": optimizar_para,
                "lineas_detectadas": num_horizontal,
                "correct_spelling": correct_spelling,
            },
        }

    except Exception as e:
        logger.error("Error en /ocr/documento_completo", exc_info=True)
        # Si ya es HTTPException (ej. timeout) la relanzamos, si no, 500
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/ocr/checkboxes")
async def ocr_con_checkboxes(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    detectar_checkboxes: bool = Form(True),
    asociar_texto: bool = Form(True),
):
    """
    Detecta checkboxes, determina si están marcados y construye pares pregunta-respuesta
    utilizando asociación espacial avanzada.
    """
    validate_file(file)
    img = await read_image(file, compress=True, max_size_mb=2.0)

    # 1. Preprocesar para mejorar detección
    processed = deskew_and_clean(img.copy())

    # 2. Detectar checkboxes
    checkboxes = detect_checkboxes(processed)

    qa_pairs = []
    # 3. Asociación avanzada con texto (si se solicita)
    if asociar_texto and checkboxes:
        # Guardar imagen procesada temporalmente para obtener datos de texto
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            tmp_path = tmp.name
        try:
            # Obtener palabras con coordenadas
            text_regions = get_text_data(tmp_path, lang, timeout=120)
            text_lines = group_words_into_lines(text_regions)
            # Construir pares pregunta-respuesta (esto también enriquece los checkboxes)
            qa_pairs = build_question_answer_pairs(
                checkboxes, text_lines, tmp_path, lang
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Tiempo de espera agotado al obtener datos de texto para asociación.",
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # 4. Extraer texto completo del documento (opcional, pero se mantiene)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name
    try:
        try:
            full_text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Tiempo de espera agotado al extraer texto completo.",
            )
        full_text = clean_text(full_text)
    finally:
        os.unlink(tmp_path)

    # 5. Estructurar el texto (horarios, días, etc.)
    structured = estructurar_texto_ocr(full_text)

    return {
        "success": True,
        "filename": file.filename,
        "num_checkboxes": len(checkboxes),
        "checkboxes": checkboxes,  # Ahora incluye campos de asociación
        "full_text": full_text,
        "texto_estructurado": structured,
        "preguntas_respuestas": qa_pairs,  # Nuevo campo
        "metadata": {"language": lang, "detectar_checkboxes": detectar_checkboxes},
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
