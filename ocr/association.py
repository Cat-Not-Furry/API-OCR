# ocr/association.py
from typing import List, Dict, Any, Optional


def asociacion_multi_nivel(
    text_lines: List[Dict], checkbox_bbox: tuple
) -> Optional[Dict]:
    """
    Retorna el texto más probable asociado a un checkbox con confianza normalizada (0-100).
    Utiliza distancia, alineación, posición relativa, tamaño de fuente y confianza OCR.

    Args:
        text_lines: Lista de diccionarios con claves 'text', 'bbox' [x, y, w, h] y opcional 'conf'.
        checkbox_bbox: Tupla (x, y, w, h) del checkbox.

    Returns:
        Diccionario con 'texto', 'confianza', 'distancia', 'bbox' o None si no hay candidatos.
    """
    x_cb, y_cb, w_cb, h_cb = checkbox_bbox
    centro_cb = (x_cb + w_cb / 2, y_cb + h_cb / 2)
    mejores = []

    for line in text_lines:
        x_t, y_t, w_t, h_t = line["bbox"]
        centro_t = (x_t + w_t / 2, y_t + h_t / 2)

        # Distancia euclidiana (menor distancia => mayor puntuación)
        dist = (
            (centro_cb[0] - centro_t[0]) ** 2 + (centro_cb[1] - centro_t[1]) ** 2
        ) ** 0.5
        dist_score = max(0, 100 - dist / 2)  # Máx 100 cuando dist=0

        # Alineación vertical (misma línea)
        align_score = 50 if abs(y_t - y_cb) < 20 else 0

        # Posición relativa (prioridad: texto a la izquierda)
        if x_t + w_t < x_cb:  # texto a la izquierda (ideal)
            pos_score = 30
        elif x_t > x_cb + w_cb:  # texto a la derecha
            pos_score = 20
        else:  # texto encima o debajo
            pos_score = 10

        # Tamaño de fuente estimado (líneas más altas suelen ser más importantes)
        font_score = min(30, h_t / 2)

        # Confianza del OCR (si existe, campo 'conf' o valor por defecto 70)
        conf_score = line.get("conf", 70) / 2  # normalizado a máximo 50

        total = dist_score + align_score + pos_score + font_score + conf_score
        norm_conf = min(100, total / 2.6)  # ajuste empírico para escala 0-100

        mejores.append(
            {
                "texto": line["text"],
                "confianza": norm_conf,
                "distancia": dist,
                "bbox": line["bbox"],
            }
        )

    mejores.sort(key=lambda x: x["confianza"], reverse=True)
    return mejores[0] if mejores else None


def agrupar_checkboxes_ordenados(
    checkboxes: List[Dict], umbral_vertical: int = 50
) -> List[List[Dict]]:
    """
    Agrupa checkboxes por proximidad vertical, respetando el orden de lectura.
    Útil para detectar grupos de opciones (radio buttons) donde solo una debería estar marcada.

    Args:
        checkboxes: Lista de diccionarios, cada uno debe tener clave 'bbox' [x, y, w, h].
        umbral_vertical: Distancia máxima en Y para considerar que están en el mismo grupo.

    Returns:
        Lista de grupos, cada grupo es una lista de checkboxes.
    """
    # Ordenar por Y (fila) y luego por X (columna)
    ordenados = sorted(checkboxes, key=lambda cb: (cb["bbox"][1], cb["bbox"][0]))
    grupos = []
    grupo_actual = []

    for cb in ordenados:
        if not grupo_actual:
            grupo_actual = [cb]
            continue
        ultimo = grupo_actual[-1]
        dist_y = abs(cb["bbox"][1] - ultimo["bbox"][1])
        if dist_y < umbral_vertical:
            grupo_actual.append(cb)
        else:
            grupos.append(grupo_actual)
            grupo_actual = [cb]
    if grupo_actual:
        grupos.append(grupo_actual)

    return grupos


# =============================================================================
# CONSTRUCCIÓN DE PARES PREGUNTA‑RESPUESTA
# =============================================================================


def build_question_answer_pairs(
    checkboxes: List[Dict],
    text_lines: List[Dict],
    form_image_path: str = None,  # opcional, se conserva por compatibilidad
    lang: str = None,
) -> List[Dict[str, Any]]:
    """
    Construye pares pregunta-respuesta a partir de checkboxes y texto.
    Utiliza asociación multi-nivel y corrige grupos de radio buttons.

    Args:
        checkboxes: Lista de checkboxes detectados (cada uno con 'bbox', 'marked', 'confidence', etc.)
        text_lines: Lista de líneas de texto (cada una con 'text', 'bbox', opcional 'conf')
        form_image_path: (no usado) se mantiene por compatibilidad
        lang: (no usado) se mantiene por compatibilidad

    Returns:
        Lista de pares pregunta-respuesta.
    """
    # 1. Asociar cada checkbox con su texto usando el algoritmo multi-nivel
    for cb in checkboxes:
        mejor_texto = asociacion_multi_nivel(text_lines, cb["bbox"])
        if mejor_texto:
            cb["associated_text"] = mejor_texto["texto"]
            cb["association_confidence"] = mejor_texto["confianza"]
        else:
            cb["associated_text"] = ""
            cb["association_confidence"] = 0.0

    # 2. Agrupar checkboxes por proximidad vertical (posibles grupos de opciones)
    grupos = agrupar_checkboxes_ordenados(checkboxes)

    # 3. Para cada grupo, si hay múltiples marcados, conservar el de mayor confianza
    for grupo in grupos:
        marcados = [cb for cb in grupo if cb.get("marked", False)]
        if len(marcados) > 1:
            # Elegir el de mayor confianza de marcado (usar 'confidence' o 0)
            mejor = max(marcados, key=lambda cb: cb.get("confidence", 0))
            for cb in grupo:
                if cb != mejor:
                    cb["marked"] = False  # desmarcar los demás
                    cb["corregido"] = True  # opcional: indicar corrección

    # 4. Construir pares pregunta-respuesta finales
    qa_pairs = []
    for cb in checkboxes:
        if cb.get("associated_text"):
            qa_pairs.append(
                {
                    "pregunta": cb["associated_text"],
                    "respuesta": "marcado" if cb.get("marked", False) else "no marcado",
                    "tipo": cb.get("tipo", "desconocido"),
                    "confianza": (
                        cb.get("confidence", 0) + cb.get("association_confidence", 0)
                    )
                    / 2,
                    "bbox_checkbox": cb["bbox"],
                    "corregido_por_grupo": cb.get("corregido", False),
                }
            )

    # 5. Opcional: detectar grupos de opciones en los QA
    qa_pairs = _detect_option_groups(qa_pairs)

    return qa_pairs


def _detect_option_groups(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Detecta grupos de opciones (pregunta: "Género", respuestas: ["M", "F"])
    basado en proximidad vertical y texto similar.
    (Implementación básica, puedes mejorarla según necesidades)
    """
    # Por ahora solo retorna la lista sin modificar
    return qa_pairs
