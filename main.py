from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io
import os
from pathlib import Path  # <-- ¡CORREGIDO! (antes era: from pathlib Path)
import logging
import cv2
import numpy as np
import re

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar rutas de Tesseract - VERSIÓN ROBUSTA
BASE_DIR = Path(__file__).parent.absolute()
logger.info(f"🔍 Directorio base: {BASE_DIR}")

# Posibles rutas para el binario de Tesseract
posibles_rutas = [
    os.path.join(BASE_DIR, "bin", "tesseract"),  # /opt/render/project/src/bin/tesseract
    os.path.join(os.getcwd(), "bin", "tesseract"),  # ./bin/tesseract
    "/opt/render/project/src/bin/tesseract",  # Ruta absoluta Render
    "./bin/tesseract",  # Ruta relativa
]

TESSERACT_PATH = None
for ruta in posibles_rutas:
    if os.path.exists(ruta):
        TESSERACT_PATH = ruta
        logger.info(f"✅ Tesseract encontrado en: {ruta}")
        # Verificar permisos
        if os.access(ruta, os.X_OK):
            logger.info(f"✅ Tiene permisos de ejecución")
        else:
            logger.warning(f"⚠️ No tiene permisos de ejecución")
        break

if not TESSERACT_PATH:
    logger.error(f"❌ Tesseract NO encontrado en ninguna ruta")
    # Listar contenido para debug
    logger.info(f"Contenido de {BASE_DIR}: {os.listdir(BASE_DIR)}")
    if os.path.exists(os.path.join(BASE_DIR, "bin")):
        logger.info(f"Contenido de bin/: {os.listdir(os.path.join(BASE_DIR, 'bin'))}")
else:
    # Configurar pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    os.environ["TESSDATA_PREFIX"] = os.path.join(BASE_DIR, "tessdata")
    logger.info(f"✅ TESSDATA_PREFIX: {os.environ['TESSDATA_PREFIX']}")

    # Verificar idiomas disponibles
    try:
        languages = pytesseract.get_languages()
        logger.info(f"✅ Idiomas disponibles: {languages}")
    except Exception as e:
        logger.error(f"❌ Error cargando idiomas: {e}")

# Crear app FastAPI
app = FastAPI(
    title="API OCR AIDA",
    description="API para extraer texto de formularios educativos",
    version="2.0.0",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Información del servicio"""
    info = {
        "message": "API OCR AIDA funcionando",
        "status": "ok",
        "tesseract_path": TESSERACT_PATH,
        "tesseract_exists": TESSERACT_PATH is not None,
        "tessdata_prefix": os.environ.get("TESSDATA_PREFIX", "No configurado"),
    }

    if TESSERACT_PATH:
        try:
            info["tesseract_version"] = str(pytesseract.get_tesseract_version())
            info["languages"] = pytesseract.get_languages()
        except Exception as e:
            info["tesseract_error"] = str(e)

    return info


@app.get("/health")
async def health_check():
    """Health check para Render"""
    return {"status": "healthy"}


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...), lang: str = "spa", psm: int = 6):
    """
    Endpoint básico para extraer texto de una imagen
    """
    try:
        # Validar archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "El archivo debe ser una imagen")

        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        logger.info(f"Procesando imagen: {file.filename}, tamaño: {image.size}")

        # Verificar que Tesseract está configurado
        if not TESSERACT_PATH:
            raise HTTPException(500, "Tesseract no está configurado correctamente")

        # Configurar Tesseract
        custom_config = f"--psm {psm} -l {lang} --oem 3"

        # Realizar OCR
        text = pytesseract.image_to_string(image, config=custom_config)

        return JSONResponse(
            {
                "success": True,
                "filename": file.filename,
                "text": text,
                "metadata": {
                    "language": lang,
                    "psm": psm,
                    "image_size": image.size,
                    "tesseract_path": TESSERACT_PATH,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        raise HTTPException(500, f"Error en OCR: {str(e)}")


# Clase para OCR perfeccionado (versión simplificada)
class OCRProcesador:
    def __init__(self):
        self.tesseract_path = TESSERACT_PATH
        self.tessdata_path = os.environ.get("TESSDATA_PREFIX")

    def preprocesar_imagen(self, image_bytes):
        """Preprocesamiento básico de imagen"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convertir a grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mejorar contraste
        gray = cv2.equalizeHist(gray)

        # Reducir ruido
        gray = cv2.medianBlur(gray, 3)

        return gray


ocr_procesador = OCRProcesador()


@app.post("/ocr/mejorado")
async def ocr_mejorado(file: UploadFile = File(...)):
    """
    Endpoint con preprocesamiento mejorado para formularios
    """
    try:
        contents = await file.read()

        # Preprocesar
        img_procesada = ocr_procesador.preprocesar_imagen(contents)

        # Convertir numpy array a PIL Image para pytesseract
        img_pil = Image.fromarray(img_procesada)

        # Configuración optimizada
        config = r"--oem 3 --psm 6 -l spa"

        # OCR
        text = pytesseract.image_to_string(img_pil, config=config)

        return JSONResponse(
            {
                "success": True,
                "filename": file.filename,
                "text": text,
                "preprocesado": True,
            }
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
