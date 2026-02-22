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
    """
    Extrae horas y rangos horarios del texto.
    Soporta formatos como 9:00 AM, 9:00 a.m., 9:00-17:00, etc.
    """
    # Patrón de hora individual (más flexible)
    patron_hora = r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?|AM|PM)?\b"
    # Patrón de rango (dos horas separadas por guión)
    patron_rango = (
        r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?|AM|PM)?\s*[-–]\s*"
        r"\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?|AM|PM)?\b"
    )
    horas = re.findall(patron_hora, texto, flags=re.IGNORECASE)
    rangos = re.findall(patron_rango, texto, flags=re.IGNORECASE)
    # Combinar y eliminar duplicados (opcional, pero puede haber solapamientos)
    return list(set(horas + rangos))


def extraer_dias(texto: str) -> List[str]:
    """
    Extrae nombres de días de la semana y combinaciones día + número (ej. 'lunes 23').
    """
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
    # Días simples
    encontrados = [dia for dia in dias_semana if dia in texto_lower]
    # Fechas con día (ej. "lunes 23")
    patron_fecha_dia = (
        r"\b(?:lunes|martes|miércoles|jueves|viernes|sábado|domingo)\s+\d{1,2}\b"
    )
    encontrados += re.findall(patron_fecha_dia, texto_lower)
    return list(set(encontrados))


def extraer_fechas(texto: str) -> List[str]:
    """
    Extrae fechas completas en formato 'dd de mes de aaaa' (ej. '23 de mayo de 2024').
    """
    patron = r"\b\d{1,2}\s+de\s+[a-zA-Záéíóúñ]+\s+de\s+\d{4}\b"
    return re.findall(patron, texto, flags=re.IGNORECASE)


def extraer_materiales(texto: str) -> List[str]:
    """
    Extrae lista de materiales mencionados después de 'deben traer'.
    """
    match = re.search(r"deben traer\s*([^\.\n]+)", texto, re.IGNORECASE)
    if match:
        items = re.split(r"[,\n]", match.group(1))
        return [i.strip() for i in items if i.strip()]
    return []


def extraer_notas(texto: str) -> str:
    """
    Extrae líneas que contienen información adicional (notas, asistencias).
    """
    lineas = texto.split("\n")
    notas = []
    for linea in lineas:
        if "asisten el día" in linea.lower() or "nota" in linea.lower():
            notas.append(linea.strip())
    return " ".join(notas)


def corregir_ortografia(texto: str, idioma: str = "es") -> str:
    """
    Corrige ortografía usando SpellChecker si está disponible.
    """
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
    texto_bruto: str, correct_spelling: bool = False
) -> Dict[str, Any]:
    """
    Función principal que orquesta la limpieza y extracción de información.
    Si correct_spelling es True, aplica corrección ortográfica.
    """
    texto_limpio = limpiar_texto_ocr(texto_bruto)
    if correct_spelling:
        texto_limpio = corregir_ortografia(texto_limpio)
    return {
        "texto_limpio": texto_limpio,
        "horarios": extraer_horarios(texto_limpio),
        "dias": extraer_dias(texto_limpio),
        "materiales": extraer_materiales(texto_limpio),
        "notas": extraer_notas(texto_limpio),
        "fechas": extraer_fechas(texto_limpio),
    }
