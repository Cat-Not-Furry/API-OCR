import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Intento de importar spellchecker (opcional)
try:
    from spellchecker import SpellChecker

    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logger.warning("Spellchecker no instalado, se omitirá corrección ortográfica.")


def clean_text(text: str) -> str:
    """Limpieza básica (espacios, saltos de línea)."""
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def limpiar_texto_ocr(texto_bruto: str) -> str:
    """
    Limpia el texto OCR con reglas contextuales (grados, decimales, etc.).
    """
    if not texto_bruto:
        return ""
    texto = texto_bruto.replace("|", " ")
    texto = texto.replace("  ", " ")
    # Corregir grados: 3%A -> 3° A
    texto = re.sub(r"(\d+)[%*]\s*([A-Za-z])", r"\1° \2", texto)
    texto = re.sub(r"(\d+)[%*]", r"\1°", texto)
    # Corregir decimales: 6,2 -> 6.2
    texto = re.sub(r"(\d+),(\d+)", r"\1.\2", texto)
    texto = " ".join(texto.split())
    return texto


def extraer_horarios(texto: str) -> List[str]:
    patron_hora = r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b"
    horas = re.findall(patron_hora, texto)
    patron_rango = r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*[-–]\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b"
    rangos = re.findall(patron_rango, texto)
    return horas + rangos


def extraer_dias(texto: str) -> List[str]:
    texto_lower = texto.lower()
    dias_semana = [
        "lunes",
        "martes",
        "miércoles",
        "jueves",
        "viernes",
        "sábado",
        "domingo",
    ]
    dias_encontrados = [dia for dia in dias_semana if dia in texto_lower]
    patron_fecha = (
        r"\b(?:lunes|martes|miércoles|jueves|viernes|sábado|domingo)\s+\d{1,2}\b"
    )
    fechas = re.findall(patron_fecha, texto_lower)
    return dias_encontrados + fechas


def extraer_materiales(texto: str) -> List[str]:
    patron = r"deben traer\s*([^\.]+)"
    match = re.search(patron, texto.lower())
    if match:
        items = match.group(1).split(",")
        return [item.strip() for item in items if item.strip()]
    return []


def extraer_notas(texto: str) -> str:
    lineas = texto.split("\n")
    notas = []
    for linea in lineas:
        if "asisten el día" in linea.lower() or "nota" in linea.lower():
            notas.append(linea.strip())
    return " ".join(notas)


def corregir_ortografia(texto: str, idioma: str = "es") -> str:
    if not SPELLCHECKER_AVAILABLE:
        return texto
    try:
        spell = SpellChecker(language=idioma)
        palabras = texto.split()
        corregidas = []
        for palabra in palabras:
            if palabra and not spell.correction(palabra) == palabra:
                corregida = spell.correction(palabra)
                corregidas.append(corregida if corregida else palabra)
            else:
                corregidas.append(palabra)
        return " ".join(corregidas)
    except Exception as e:
        logger.error(f"Error en corrección ortográfica: {e}")
        return texto


def estructurar_texto_ocr(
    texto_bruto: str, corregir_ortografia_flag: bool = False
) -> dict:
    texto_limpio = limpiar_texto_ocr(texto_bruto)
    if corregir_ortografia_flag:
        texto_limpio = corregir_ortografia(texto_limpio)
    return {
        "texto_limpio": texto_limpio,
        "horarios": extraer_horarios(texto_limpio),
        "dias": extraer_dias(texto_limpio),
        "materiales": extraer_materiales(texto_limpio),
        "notas": extraer_notas(texto_limpio),
    }
