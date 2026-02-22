# ocr/__init__.py
from .engine import run_tesseract, ocr_region
from .postprocess import clean_text, extract_entities

__all__ = ["run_tesseract", "ocr_region", "clean_text", "extract_entities"]
