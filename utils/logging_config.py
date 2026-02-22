# utils/logging_config.py
import logging
import sys


def setup_logging(level=logging.INFO):
    """Configura el logging para toda la aplicación."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)
