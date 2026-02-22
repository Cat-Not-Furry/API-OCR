# background.py
import uuid
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Almacenamiento en memoria (para pruebas, en producción usar Redis/BD)
_task_results: Dict[str, Dict[str, Any]] = {}
_task_status: Dict[str, str] = {}  # "pending", "processing", "done", "error"


def create_task() -> str:
    """Crea una nueva tarea y devuelve su ID."""
    task_id = str(uuid.uuid4())
    _task_status[task_id] = "pending"
    _task_results[task_id] = {}
    logger.info(f"Tarea creada: {task_id}")
    return task_id


def update_task(task_id: str, status: str, result: Optional[Dict] = None):
    """Actualiza el estado y resultado de una tarea."""
    _task_status[task_id] = status
    if result:
        _task_results[task_id] = result
    logger.info(f"Tarea {task_id} actualizada a {status}")


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Obtiene el estado y resultado de una tarea."""
    if task_id not in _task_status:
        return None
    return {
        "task_id": task_id,
        "status": _task_status[task_id],
        "result": _task_results.get(task_id, {}),
    }


def clean_old_tasks(max_age_seconds: int = 3600):
    """Limpia tareas antiguas (para llamar periódicamente)."""
    # No implementado por simplicidad, pero se puede añadir un cron.
    pass
