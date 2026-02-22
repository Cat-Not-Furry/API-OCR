# preprocessing/__init__.py
from .enhance import (
    correct_skew,
    remove_noise,
    remove_shadows,
    binarize,
    deskew_and_clean,
)
from .detection import detect_tables, extract_table_cells, segment_regions
from .compression import ImageCompressor, compress_image  # <-- NUEVO

__all__ = [
    "correct_skew",
    "remove_noise",
    "remove_shadows",
    "binarize",
    "deskew_and_clean",
    "detect_tables",
    "extract_table_cells",
    "segment_regions",
    "ImageCompressor",
    "compress_image",
]
