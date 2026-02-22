# main.py (dentro de algún endpoint, por ejemplo /ocr/checkboxes)
from integration.infinityfree import InfinityFreeClient

# Configurar cliente (puedes poner la URL en config.py)
INFINITY_URL = os.getenv(
    "INFINITY_URL", "https://keydash-user-admin.wuaze.com/other_test.html"
)
INFINITY_API_KEY = os.getenv("INFINITY_API_KEY", "")
infinity_client = InfinityFreeClient(INFINITY_URL, INFINITY_API_KEY)


@app.post("/ocr/checkboxes")
async def ocr_con_checkboxes(
    file: UploadFile = File(...),
    lang: str = Form(DEFAULT_LANG),
    detectar_checkboxes: bool = Form(True),
    asociar_texto: bool = Form(True),
    enviar_a_infinity: bool = Form(False),  # <-- nuevo parámetro
):
    # ... (código de procesamiento existente) ...

    result = {
        "success": True,
        "filename": file.filename,
        "num_checkboxes": len(checkboxes),
        "checkboxes": checkboxes,
        "full_text": full_text,
        "texto_estructurado": structured,
        "metadata": {"language": lang, "detectar_checkboxes": detectar_checkboxes},
    }

    if enviar_a_infinity:
        try:
            # Enviar checkboxes y texto
            infinity_client.send_checkboxes(
                checkboxes, file.filename, result["metadata"]
            )
            infinity_client.send_text(full_text, structured, file.filename)
        except Exception as e:
            logger.error(f"Error enviando a InfinityFree: {e}")
            # No interrumpimos la respuesta, solo logueamos

    return result
