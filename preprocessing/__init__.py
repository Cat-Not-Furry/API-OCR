# preprocessing/__init__.py
from .enhance import (
    correct_skew,
    remove_noise,
    remove_shadows,
    binarize,
    deskew_and_clean,
    detect_document_contour,
)
from .detection import detect_tables, extract_table_cells, segment_regions
from .compression import ImageCompressor, compress_image
from .checkbox import detect_checkboxes, associate_checkboxes_with_text  # <-- NUEVO

__all__ = [
    "correct_skew",
    "remove_noise",
    "remove_shadows",
    "binarize",
    "deskew_and_clean",
    "detect_document_contour",
    "detect_tables",
    "extract_table_cells",
    "segment_regions",
    "ImageCompressor",
    "compress_image",
    "detect_checkboxes",
    "associate_checkboxes_with_text",
]
