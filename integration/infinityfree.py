import os
import requests
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class InfinityFreeClient:
    """Cliente para enviar resultados a InfinityFree."""

    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    async def send_ocr_result(
        self, filename: str, text: str, metadata: Dict[str, Any]
    ) -> bool:
        try:
            payload = {
                "filename": filename,
                "text": text,
                "metadata": metadata,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
            response = self.session.post(
                f"{self.base_url}/callback.php", json=payload, timeout=30
            )
            if response.status_code == 200:
                logger.info(f"Resultado enviado exitosamente para {filename}")
                return True
            else:
                logger.error(
                    f"Error enviando resultado: {response.status_code} - {response.text}"
                )
                return False
        except Exception as e:
            logger.error(f"Error en envío a InfinityFree: {e}")
            return False
