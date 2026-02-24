# ocr/association.py
from preprocessing.checkbox import associate_checkboxes_with_text_advanced
from typing import List, Dict, Any


def build_question_answer_pairs(
    checkboxes: List[Dict],
    text_lines: List[Dict],
    form_image_path: str = None,  # opcional, para debugging
    lang: str = None,
) -> List[Dict[str, Any]]:
    """
    Construye pares pregunta-respuesta a partir de checkboxes y texto.
    """
    # 1. Asociar checkboxes con texto
    associated = associate_checkboxes_with_text_advanced(checkboxes, text_lines)

    # 2. Agrupar por pregunta (texto asociado)
    qa_pairs = []
    for cb in associated:
        if cb["associated_text"]:
            qa_pairs.append(
                {
                    "pregunta": cb["associated_text"],
                    "respuesta": "marcado" if cb["marked"] else "no marcado",
                    "tipo": cb["tipo"],
                    "confianza": (
                        cb["confidence"] + cb.get("association_confidence", 0)
                    )
                    / 2,
                    "bbox_checkbox": cb["bbox"],
                }
            )

    # 3. Opcional: detectar grupos de opciones
    qa_pairs = _detect_option_groups(qa_pairs)

    return qa_pairs


def _detect_option_groups(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Detecta grupos de opciones (pregunta: "Género", respuestas: ["M", "F"])
    basado en proximidad vertical y texto similar.
    """
    # Implementación: agrupar por coordenada Y cercana y texto común
    # ... (puedes expandir según necesites)
    return qa_pairs
