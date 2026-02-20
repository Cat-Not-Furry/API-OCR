# main.py
import os
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import subprocess

# Usar el binario incluido
TESSERACT_PATH = os.path.join(os.path.dirname(__file__), "bin", "tesseract")
os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(__file__), "tessdata")

# Configurar pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AIDA OCR API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringe a tu dominio InfinityFree
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Tesseract (Render lo tiene en /usr/bin)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


@app.get("/")
async def root():
    """Endpoint de verificación"""
    return {
        "service": "AIDA OCR API",
        "status": "online",
        "python_version": "3.12",
        "tesseract_version": str(pytesseract.get_tesseract_version()),
    }


@app.get("/health")
async def health():
    """Health check para Render"""
    return {"status": "healthy"}


@app.post("/ocr")
async def procesar_ocr(file: UploadFile = File(...)):
    """
    Recibe imagen, ejecuta OCR y devuelve JSON
    """
    logger.info(f"Procesando archivo: {file.filename}")

    try:
        # Validar tipo de archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "Solo se aceptan imágenes")

        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(400, "No se pudo decodificar la imagen")

        # PREPROCESAMIENTO (OpenCV moderno)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR con Tesseract 5 (mucho mejor que v3)
        custom_config = r"--oem 3 --psm 6 -l spa"
        text = pytesseract.image_to_string(binary, config=custom_config)

        # Detectar checkboxes
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        checkboxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 15 < w < 50 and 15 < h < 50:
                # Verificar si es cuadrado
                aspect = w / h
                if 0.8 < aspect < 1.2:
                    # Determinar si está marcado
                    roi = binary[y : y + h, x : x + w]
                    fill_ratio = cv2.countNonZero(roi) / (w * h)

                    checkboxes.append(
                        {
                            "x": int(x),
                            "y": int(y),
                            "w": int(w),
                            "h": int(h),
                            "marcado": fill_ratio > 0.3,
                        }
                    )

        resultado = {
            "filename": file.filename,
            "texto_completo": text,
            "checkboxes": checkboxes[:20],  # Límite por si acaso
            "estadisticas": {
                "caracteres": len(text),
                "checkboxes_detectados": len(checkboxes),
            },
        }

        logger.info(
            f"Procesado exitoso: {len(text)} caracteres, {len(checkboxes)} checkboxes"
        )
        return resultado

    except Exception as e:
        logger.error(f"Error procesando {file.filename}: {str(e)}")
        raise HTTPException(500, f"Error en OCR: {str(e)}")


# Para ejecución local
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
