# config.py
import os
from pathlib import Path

# Rutas base
BASE_DIR = Path(__file__).parent.absolute()
TESSERACT_PATH = os.path.join(BASE_DIR, "bin", "tesseract")
TESSDATA_PATH = os.path.join(BASE_DIR, "tessdata")
INFINITYFREE_URL = os.getenv(
    "INFINITYFREE_URL", "http://default-domain.infinityfreeapp.com/other_test.html"
)

# Límites
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"}

# Configuración de OCR por defecto
DEFAULT_LANG = "spa"
DEFAULT_PSM = 6
DEFAULT_OEM = 3
