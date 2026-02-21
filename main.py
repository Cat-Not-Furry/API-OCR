import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import subprocess
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Obtener rutas absolutas
BASE_DIR = Path(__file__).parent.absolute()
TESSERACT_BIN = os.path.join(BASE_DIR, "bin", "tesseract")
TESSDATA_DIR = os.path.join(BASE_DIR, "tessdata")

logger.info(f"🔍 Directorio base: {BASE_DIR}")
logger.info(f"🔍 Tesseract bin: {TESSERACT_BIN}")
logger.info(f"🔍 Tessdata dir: {TESSDATA_DIR}")

# Verificar que el binario existe y es ejecutable
if not os.path.exists(TESSERACT_BIN):
    logger.error(f"❌ No existe el binario en: {TESSERACT_BIN}")
    # Listar contenido de bin/
    if os.path.exists(os.path.join(BASE_DIR, "bin")):
        logger.info(f"Contenido de bin/: {os.listdir(os.path.join(BASE_DIR, 'bin'))}")
else:
    logger.info(f"✅ Binario encontrado")
    if os.access(TESSERACT_BIN, os.X_OK):
        logger.info(f"✅ Permisos de ejecución OK")
    else:
        logger.warning(f"⚠️ Sin permisos de ejecución, intentando arreglar...")
        try:
            os.chmod(TESSERACT_BIN, 0o755)
            logger.info(f"✅ Permisos corregidos")
        except Exception as e:
            logger.error(f"❌ No se pudo cambiar permisos: {e}")

# Verificar tessdata
if os.path.exists(TESSDATA_DIR):
    logger.info(f"✅ Tessdata encontrado")
    traineddata_files = [
        f for f in os.listdir(TESSDATA_DIR) if f.endswith(".traineddata")
    ]
    logger.info(f"Archivos traineddata: {traineddata_files}")
else:
    logger.error(f"❌ No existe tessdata en: {TESSDATA_DIR}")


# --- SOLUCIÓN: Función wrapper que usa subprocess directamente ---
def tesseract_ocr(image_path, lang="spa", psm=6):
    """
    Ejecuta Tesseract directamente usando subprocess
    Esto evita cualquier problema con pytesseract
    """
    try:
        # Construir comando
        cmd = [
            TESSERACT_BIN,
            image_path,
            "stdout",  # Salida a stdout
            "-l",
            lang,
            "--psm",
            str(psm),
            "--oem",
            "3",
        ]

        # Configurar entorno
        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = TESSDATA_DIR

        logger.info(f"Ejecutando: {' '.join(cmd)}")

        # Ejecutar Tesseract
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            logger.error(f"Error de Tesseract: {result.stderr}")
            return None, result.stderr

        return result.stdout.strip(), None

    except subprocess.TimeoutExpired:
        return None, "Timeout en Tesseract"
    except Exception as e:
        return None, str(e)


def tesseract_version():
    """Obtiene la versión de Tesseract"""
    try:
        result = subprocess.run(
            [TESSERACT_BIN, "--version"], env=os.environ, capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.split("\n")[0]
    except:
        pass
    return "Desconocido"


def tesseract_languages():
    """Obtiene los idiomas disponibles"""
    try:
        result = subprocess.run(
            [TESSERACT_BIN, "--list-langs"],
            env={"TESSDATA_PREFIX": TESSDATA_DIR},
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # La primera línea es "List of available languages"
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                return lines[1:]  # Saltar la primera línea
    except:
        pass
    return []


# --- FastAPI App ---
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
    version = tesseract_version()
    languages = tesseract_languages()

    return {
        "message": "API OCR AIDA funcionando",
        "status": "ok",
        "tesseract": {
            "path": TESSERACT_BIN,
            "exists": os.path.exists(TESSERACT_BIN),
            "executable": os.access(TESSERACT_BIN, os.X_OK)
            if os.path.exists(TESSERACT_BIN)
            else False,
            "version": version,
        },
        "tessdata": {
            "path": TESSDATA_DIR,
            "exists": os.path.exists(TESSDATA_DIR),
            "languages": languages,
            "files": [f for f in os.listdir(TESSDATA_DIR) if f.endswith(".traineddata")]
            if os.path.exists(TESSDATA_DIR)
            else [],
        },
    }


@app.get("/health")
async def health_check():
    """Health check para Render"""
    return {"status": "healthy"}


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...), lang: str = "spa", psm: int = 6):
    """
    Endpoint para extraer texto de una imagen usando Tesseract directamente
    """
    temp_path = None
    try:
        # Validar archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "El archivo debe ser una imagen")

        # Guardar imagen temporalmente
        contents = await file.read()

        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        logger.info(
            f"Procesando imagen: {file.filename}, tamaño: {len(contents)} bytes"
        )

        # Ejecutar Tesseract
        text, error = tesseract_ocr(temp_path, lang=lang, psm=psm)

        if error:
            raise HTTPException(500, f"Error en OCR: {error}")

        return JSONResponse(
            {
                "success": True,
                "filename": file.filename,
                "text": text or "",
                "metadata": {
                    "language": lang,
                    "psm": psm,
                    "tesseract_path": TESSERACT_BIN,
                    "tessdata_path": TESSDATA_DIR,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        raise HTTPException(500, f"Error en OCR: {str(e)}")

    finally:
        # Limpiar archivo temporal
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/ocr/debug")
async def ocr_debug(file: UploadFile = File(...)):
    """
    Endpoint de debug que muestra información detallada
    """
    result = {
        "filename": file.filename,
        "content_type": file.content_type,
        "tesseract": {
            "path": TESSERACT_BIN,
            "exists": os.path.exists(TESSERACT_BIN),
            "executable": os.access(TESSERACT_BIN, os.X_OK)
            if os.path.exists(TESSERACT_BIN)
            else False,
            "version": tesseract_version(),
        },
        "tessdata": {
            "path": TESSDATA_DIR,
            "exists": os.path.exists(TESSDATA_DIR),
            "files": [],
        },
    }

    if os.path.exists(TESSDATA_DIR):
        result["tessdata"]["files"] = os.listdir(TESSDATA_DIR)

    # Probar ejecución básica
    try:
        version_output = subprocess.run(
            [TESSERACT_BIN, "--version"],
            env={"TESSDATA_PREFIX": TESSDATA_DIR},
            capture_output=True,
            text=True,
        )
        result["test_version"] = {
            "stdout": version_output.stdout,
            "stderr": version_output.stderr,
            "returncode": version_output.returncode,
        }
    except Exception as e:
        result["test_version_error"] = str(e)

    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
