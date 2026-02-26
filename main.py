# main.py
import os
import cv2
import logging
import subprocess
import asyncio
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tempfile
from typing import List, Dict, Tuple
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importaciones de nuestros módulos
from config import TESSERACT_PATH, TESSDATA_PATH, DEFAULT_LANG, INFINITYFREE_URL
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
from ocr.engine import run_tesseract, ocr_region, get_text_data, group_words_into_lines
from ocr.postprocess import clean_text, estructurar_texto_ocr
from preprocessing.checkbox import detect_checkboxes
from ocr.association import build_question_answer_pairs
from integration.infinityfree import InfinityFreeClient
from background import create_task, update_task, get_task
from metrics import OCRMetrics

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# Inicializar cliente InfinityFree
infinity_client = InfinityFreeClient(INFINITYFREE_URL)

# Inicializar métricas
metrics = OCRMetrics()

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


# ==================== FUNCIONES AUXILIARES (síncronas para background) ====================


def read_image_from_bytes(
    file_bytes: bytes, compress: bool = True, max_size_mb: float = 2.0
) -> np.ndarray:
    """
    Decodifica una imagen desde bytes y opcionalmente la redimensiona si supera max_size_mb.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Formato de imagen no válido o datos corruptos")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if compress:
        h, w = img_rgb.shape[:2]
        current_size_mb = (h * w * 3) / (1024 * 1024)
        if current_size_mb > max_size_mb:
            scale = (max_size_mb / current_size_mb) ** 0.5
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_rgb


def sync_procesar_con_preprocesamiento(
    img: np.ndarray, lang: str, correccion_skew: bool, metodo_binarizacion: str
) -> Dict:
    """Versión síncrona de procesar_con_preprocesamiento (sin excepciones HTTP)."""
    try:
        if correccion_skew:
            img = correct_skew(img)
        img = remove_shadows(img)
        img = remove_noise(img, method="nlmeans")
        img = binarize(img, method=metodo_binarizacion)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        try:
            text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
            text = clean_text(text)
        finally:
            os.unlink(tmp_path)
        return {"text": text}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout en preprocesamiento"}
    except Exception as e:
        return {"error": str(e)}


def sync_procesar_con_segmentacion(
    img: np.ndarray, lang: str, detectar_tablas: bool
) -> Dict:
    """
    Versión síncrona de procesar_con_segmentacion con paralelización usando ThreadPoolExecutor.
    """
    try:
        processed = deskew_and_clean(img.copy())
        regions = segment_regions(processed)

        if detectar_tablas:
            tables = detect_tables(processed)
            for table in tables:
                contained = False
                for reg in regions:
                    rx, ry, rw, rh = reg["bbox"]
                    tx, ty, tw, th = table["bbox"]
                    if (
                        rx <= tx
                        and ry <= ty
                        and rx + rw >= tx + tw
                        and ry + rh >= ty + th
                    ):
                        contained = True
                        break
                if not contained:
                    regions.append(table)

        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        resultados = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {
                executor.submit(ocr_region, img, reg, lang, 120): i
                for i, reg in enumerate(regions)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                reg = regions[i]
                try:
                    texto = future.result()
                    resultados.append(
                        {
                            "region": i,
                            "tipo": reg["type"],
                            "bbox": reg["bbox"],
                            "texto": texto,
                        }
                    )
                except subprocess.TimeoutExpired:
                    resultados.append(
                        {
                            "region": i,
                            "tipo": reg["type"],
                            "bbox": reg["bbox"],
                            "error": "Timeout",
                        }
                    )
                except Exception as e:
                    logger.error(f"Error en región {i}: {e}")
                    resultados.append(
                        {
                            "region": i,
                            "tipo": reg["type"],
                            "bbox": reg["bbox"],
                            "error": str(e),
                        }
                    )

        resultados.sort(key=lambda x: x["region"])
        texto_completo = "\n".join([r["texto"] for r in resultados if "texto" in r])
        return {
            "num_regiones": len(resultados),
            "texto_completo": texto_completo,
            "regiones": resultados,
        }
    except Exception as e:
        return {"error": str(e)}


def sync_procesar_como_tabla(img: np.ndarray, lang: str) -> Dict:
    """Versión síncrona de procesar_como_tabla."""
    try:
        processed = deskew_and_clean(img)
        tables = detect_tables(processed)

        if not tables:
            return {"error": "No se detectaron tablas"}

        main_table = max(tables, key=lambda t: t["bbox"][2] * t["bbox"][3])
        bbox = main_table["bbox"]
        roi = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
            tmp_path = tmp.name
        try:
            text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
        finally:
            os.unlink(tmp_path)

        return {"tabla_texto": text, "bbox": bbox}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout procesando tabla"}
    except Exception as e:
        return {"error": str(e)}


def sync_ocr_pipeline(
    file_bytes: bytes,
    filename: str,
    lang: str,
    optimizar_para: str,
    correct_spelling: bool,
) -> Dict:
    """
    Ejecuta el pipeline completo de OCR (versión síncrona) y devuelve el resultado.
    Además registra métricas automáticamente.
    """
    original_size = len(file_bytes)
    start_time = time.time()
    try:
        img = read_image_from_bytes(file_bytes, compress=True, max_size_mb=2.0)

        # Calcular tamaño comprimido aproximado
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        compressed_size = len(buffer)

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

        # Elegir método según el análisis o el parámetro
        if num_horizontal > 10 or optimizar_para == "tablas":
            resultado = sync_procesar_como_tabla(img, lang)
            if "error" in resultado:
                resultado = sync_procesar_con_segmentacion(
                    img, lang, detectar_tablas=True
                )
        elif optimizar_para == "mixto":
            resultado = sync_procesar_con_segmentacion(img, lang, detectar_tablas=True)
        else:
            resultado = sync_procesar_con_preprocesamiento(
                img, lang, correccion_skew=True, metodo_binarizacion="sauvola"
            )

        # Extraer texto según el tipo de resultado
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

        duration = time.time() - start_time

        # Registrar métricas
        metrics.log_request(
            {
                "filename": filename,
                "endpoint": "/ocr/async",  # Se marca como async aunque sea el pipeline interno
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "num_regiones": resultado.get("num_regiones", 0),
                "num_checkboxes": 0,
                "checkboxes_asociados": 0,
                "avg_association_conf": 0,
                "success": True,
                "metadata": {
                    "language": lang,
                    "optimizacion": optimizar_para,
                    "lineas_detectadas": num_horizontal,
                    "correct_spelling": correct_spelling,
                },
            }
        )

        return {
            "success": True,
            "filename": filename,
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
        duration = time.time() - start_time
        metrics.log_request(
            {
                "filename": filename,
                "endpoint": "/ocr/async",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": 0,
                "success": False,
                "error": str(e),
                "metadata": {
                    "language": lang,
                    "optimizacion": optimizar_para,
                    "correct_spelling": correct_spelling,
                },
            }
        )
        return {"success": False, "error": str(e)}


async def run_ocr_background(
    task_id: str,
    file_bytes: bytes,
    filename: str,
    lang: str,
    optimizar_para: str,
    correct_spelling: bool,
):
    """
    Ejecuta el OCR en un hilo separado y actualiza el estado de la tarea.
    """
    try:
        update_task(task_id, "processing")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            sync_ocr_pipeline,
            file_bytes,
            filename,
            lang,
            optimizar_para,
            correct_spelling,
        )
        update_task(task_id, "done", result)
    except Exception as e:
        logger.exception(f"Error en tarea {task_id}")
        update_task(task_id, "error", {"error": str(e)})


# ==================== FUNCIONES AUXILIARES (originales, asíncronas) ====================


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
    """
    Versión asíncrona de procesar_con_segmentacion con paralelización usando asyncio.
    """
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

    sem = asyncio.Semaphore(5)

async def procesar_una(reg, idx):
        async with sem:
            try:
                texto = await asyncio.to_thread(ocr_region, img, reg, lang, 120)
                return {
                    "region": idx,
                    "tipo": reg["type"],
                    "bbox": reg["bbox"],
                    "texto": texto,
                }
            except subprocess.TimeoutExpired:
                return {
                    "region": idx,
                    "tipo": reg["type"],
                    "bbox": reg["bbox"],
                    "error": "Timeout",
                }
            except Exception as e:
                logger.error(f"Error en región {idx}: {e}")
                return {
                    "region": idx,
                    "tipo": reg["type"],
                    "bbox": reg["bbox"],
                    "error": str(e),
                }

    tareas = [procesar_una(reg, i) for i, reg in enumerate(regions)]
    resultados = await asyncio.gather(*tareas)
    resultados.sort(key=lambda x: x["region"])

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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name
    try:
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
    start_time = time.time()
    original_size = 0
    compressed_size = 0
    try:
        validate_file(file)
        img, original_size = await read_image(file, compress=True, max_size_mb=2.0)

        # Estimar tamaño comprimido
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        compressed_size = len(buffer)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            tmp_path = tmp.name

        try:
            text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
            text = clean_text(text)
            estructurado = estructurar_texto_ocr(
                text, corregir_ortografia_flag=correct_spelling
            )
        finally:
            os.unlink(tmp_path)

        duration = time.time() - start_time

        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/basico",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "success": True,
                "metadata": {"language": lang, "correct_spelling": correct_spelling},
            }
        )

        return {
            "success": True,
            "filename": file.filename,
            "text": text,
            "texto_estructurado": estructurado,
            "metadata": {"language": lang, "correct_spelling": correct_spelling},
        }
    except Exception as e:
        duration = time.time() - start_time
        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/basico",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "success": False,
                "error": str(e),
                "metadata": {"language": lang, "correct_spelling": correct_spelling},
            }
        )
        raise


@app.post("/ocr/segmentado")
async def ocr_con_segmentacion(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    detectar_tablas: bool = Form(True),
    correct_spelling: bool = Form(False),
):
    start_time = time.time()
    original_size = 0
    compressed_size = 0
    try:
        validate_file(file)
        img, original_size = await read_image(file, compress=True, max_size_mb=2.0)

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        compressed_size = len(buffer)

        resultado = await procesar_con_segmentacion(img, lang, detectar_tablas)

        texto_completo = resultado["texto_completo"]
        estructurado = estructurar_texto_ocr(
            texto_completo, corregir_ortografia_flag=correct_spelling
        )

        duration = time.time() - start_time

        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/segmentado",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "num_regiones": resultado.get("num_regiones", 0),
                "success": True,
                "metadata": {
                    "language": lang,
                    "detectar_tablas": detectar_tablas,
                    "correct_spelling": correct_spelling,
                },
            }
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
    except Exception as e:
        duration = time.time() - start_time
        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/segmentado",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "success": False,
                "error": str(e),
                "metadata": {
                    "language": lang,
                    "detectar_tablas": detectar_tablas,
                    "correct_spelling": correct_spelling,
                },
            }
        )
        raise


@app.post("/ocr/tabla")
async def ocr_tabla(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    formato_salida: str = Form("json"),
    correct_spelling: bool = Form(False),
):
    start_time = time.time()
    original_size = 0
    compressed_size = 0
    try:
        validate_file(file)
        img, original_size = await read_image(file, compress=True, max_size_mb=2.0)

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        compressed_size = len(buffer)

        resultado = await procesar_como_tabla(img, lang)
        if "error" in resultado:
            return {"success": False, "error": resultado["error"]}

        tabla_texto = resultado["tabla_texto"]
        estructurado = estructurar_texto_ocr(
            tabla_texto, corregir_ortografia_flag=correct_spelling
        )

        duration = time.time() - start_time

        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/tabla",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "success": True,
                "metadata": {
                    "language": lang,
                    "formato_salida": formato_salida,
                    "correct_spelling": correct_spelling,
                },
            }
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
    except Exception as e:
        duration = time.time() - start_time
        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/tabla",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "success": False,
                "error": str(e),
                "metadata": {
                    "language": lang,
                    "formato_salida": formato_salida,
                    "correct_spelling": correct_spelling,
                },
            }
        )
        raise


# Función auxiliar (opcional, para evitar duplicación)
async def obtener_texto_y_coordenadas(img: np.ndarray, lang: str) -> Dict:
    """
    Ejecuta OCR sobre la imagen y retorna tanto el texto completo como las coordenadas de palabras.
    (No se usa directamente en el endpoint, pero puede servir para otros casos)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        tmp_path = tmp.name
    try:
        text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
        coords = get_text_data(tmp_path, lang, psm=3, timeout=120)
        text = clean_text(text)
        return {"text": text, "coords": coords}
    finally:
        os.unlink(tmp_path)


@app.post("/ocr/documento_completo")
async def ocr_documento_completo(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    optimizar_para: str = Form("texto"),
    correct_spelling: bool = Form(False),
    return_coords: bool = Form(False),  # <-- NUEVO FLAG
):
    start_time = time.time()
    original_size = 0
    compressed_size = 0
    try:
        validate_file(file)
        img, original_size = await read_image(file, compress=True, max_size_mb=2.0)

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        compressed_size = len(buffer)

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

        # Elegir estrategia de procesamiento
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

        # Extraer texto del resultado
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

        # Si se solicitan coordenadas, obtenerlas (sobre la imagen original)
        coords_data = None
        if return_coords:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                tmp_path = tmp.name
            try:
                # Ejecutar get_text_data en un hilo para no bloquear
                coords_data = await asyncio.to_thread(
                    get_text_data, tmp_path, lang, 3, 120
                )
            finally:
                os.unlink(tmp_path)

        duration = time.time() - start_time

        # Registrar métricas
        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/documento_completo",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "num_regiones": resultado.get("num_regiones", 0),
                "success": True,
                "metadata": {
                    "language": lang,
                    "optimizacion": optimizar_para,
                    "lineas_detectadas": num_horizontal,
                    "correct_spelling": correct_spelling,
                    "return_coords": return_coords,
                },
            }
        )

        # Construir respuesta
        response = {
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
        if return_coords and coords_data:
            response["coordenadas"] = coords_data

        return response

    except Exception as e:
        duration = time.time() - start_time
        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/documento_completo",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "success": False,
                "error": str(e),
                "metadata": {
                    "language": lang,
                    "optimizacion": optimizar_para,
                    "correct_spelling": correct_spelling,
                },
            }
        )
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/ocr/checkboxes")
async def ocr_con_checkboxes(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    detectar_checkboxes: bool = Form(True),
    asociar_texto: bool = Form(True),
    return_coords: bool = Form(False),  # <-- NUEVO FLAG
):
    start_time = time.time()
    original_size = 0
    compressed_size = 0
    try:
        validate_file(file)
        img, original_size = await read_image(file, compress=True, max_size_mb=2.0)

        # Calcular tamaño comprimido aproximado
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        compressed_size = len(buffer)

        processed = deskew_and_clean(img.copy())
        checkboxes = detect_checkboxes(processed)

        qa_pairs = []
        checkboxes_asociados = 0
        avg_conf = 0
        coordenadas = []  # Para almacenar coordenadas de palabras

        # Si necesitamos texto para asociación o coordenadas, obtenemos datos
        if asociar_texto or return_coords:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                cv2.imwrite(tmp.name, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                tmp_path = tmp.name
            try:
                text_regions = get_text_data(tmp_path, lang, timeout=120)
                text_lines = group_words_into_lines(text_regions)

                if return_coords:
                    # Convertir a formato JSON serializable
                    coordenadas = [
                        {
                            "texto": w["text"],
                            "bbox": list(w["bbox"]),
                            "confianza": w["conf"],
                            "linea": w["line"],
                            "bloque": w["block"],
                            "parrafo": w["par"],
                        }
                        for w in text_regions
                    ]

                if asociar_texto and checkboxes:
                    qa_pairs = build_question_answer_pairs(
                        checkboxes, text_lines, tmp_path, lang
                    )
                    checkboxes_asociados = len(qa_pairs)
                    if qa_pairs:
                        avg_conf = sum(p.get("confianza", 0) for p in qa_pairs) / len(
                            qa_pairs
                        )
            except subprocess.TimeoutExpired:
                raise HTTPException(
                    status_code=504,
                    detail="Tiempo de espera agotado al obtener datos de texto.",
                )
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Extraer texto completo del documento
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            tmp_path = tmp.name
        try:
            full_text = run_tesseract(tmp_path, lang, psm=6, timeout=120)
            full_text = clean_text(full_text)
        finally:
            os.unlink(tmp_path)

        structured = estructurar_texto_ocr(full_text)
        duration = time.time() - start_time

        # Registrar métricas
        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/checkboxes",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "num_checkboxes": len(checkboxes),
                "checkboxes_asociados": checkboxes_asociados,
                "avg_association_conf": avg_conf,
                "success": True,
                "metadata": {
                    "language": lang,
                    "detectar_checkboxes": detectar_checkboxes,
                    "asociar_texto": asociar_texto,
                    "return_coords": return_coords,
                },
            }
        )

        # Construir respuesta
        response = {
            "success": True,
            "filename": file.filename,
            "num_checkboxes": len(checkboxes),
            "checkboxes": checkboxes,
            "full_text": full_text,
            "texto_estructurado": structured,
            "preguntas_respuestas": qa_pairs,
            "metadata": {
                "language": lang,
                "detectar_checkboxes": detectar_checkboxes,
            },
        }
        if return_coords:
            response["coordenadas"] = coordenadas

        return response

    except Exception as e:
        duration = time.time() - start_time
        metrics.log_request(
            {
                "filename": file.filename,
                "endpoint": "/ocr/checkboxes",
                "duration": duration,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "success": False,
                "error": str(e),
                "metadata": {
                    "language": lang,
                    "detectar_checkboxes": detectar_checkboxes,
                    "asociar_texto": asociar_texto,
                    "return_coords": return_coords,
                },
            }
        )
        raise


@app.post("/ocr/async")
async def ocr_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    optimizar_para: str = Form("texto"),
    correct_spelling: bool = Form(False),
):
    """
    Endpoint asíncrono para imágenes grandes.
    Si el archivo supera 5 MB, se crea una tarea y se procesa en segundo plano.
    Para archivos más pequeños, se procesa de forma síncrona (misma respuesta inmediata).
    """
    validate_file(file)
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb <= 5:
        # Procesar directamente (síncrono, pero dentro de un hilo para no bloquear)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            sync_ocr_pipeline,
            contents,
            file.filename,
            lang,
            optimizar_para,
            correct_spelling,
        )
        # Devolver el resultado directamente (ya contiene 'success')
        return {**result, "async": False}
    else:
        task_id = create_task()
        asyncio.create_task(
            run_ocr_background(
                task_id, contents, file.filename, lang, optimizar_para, correct_spelling
            )
        )
        return {"task_id": task_id, "status": "pending", "async": True}


@app.get("/ocr/result/{task_id}")
async def get_ocr_result(task_id: str):
    """
    Consulta el estado y resultado de una tarea asíncrona.
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    return task


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
