# ocr/postprocess.py
import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Limpia el texto extraído (elimina espacios extras, normaliza)."""
    # Eliminar líneas vacías múltiples
    text = re.sub(r"\n\s*\n", "\n", text)
    # Eliminar espacios múltiples
    text = re.sub(r" +", " ", text)
    return text.strip()


def extract_entities(text: str, entity_type: str = "all") -> Dict[str, List[str]]:
    """Extrae entidades como fechas, nombres, etc. (para futuro)."""
    entities = {}

    if entity_type in ["all", "dates"]:
        # Patrón simple para fechas (DD/MM/YYYY, DD-MM-YYYY)
        dates = re.findall(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", text)
        entities["dates"] = dates

    if entity_type in ["all", "emails"]:
        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
        )
        entities["emails"] = emails

    return entities
