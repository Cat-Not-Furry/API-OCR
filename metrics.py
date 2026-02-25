# metrics.py
import sqlite3
import json
from datetime import datetime


class OCRMetrics:
    """
    Clase para registrar métricas de las solicitudes OCR en una base SQLite.
    """

    def __init__(self, db_path="ocr_metrics.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ocr_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                filename TEXT,
                endpoint TEXT,
                duration REAL,
                original_size INTEGER,
                compressed_size INTEGER,
                num_regiones INTEGER,
                num_checkboxes INTEGER,
                checkboxes_asociados INTEGER,
                avg_association_conf REAL,
                success BOOLEAN,
                error TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def log_request(self, data):
        """
        Registra una solicitud en la base de datos.
        data debe ser un diccionario con las claves esperadas.
        """
        self.conn.execute(
            """
            INSERT INTO ocr_requests (
                timestamp, filename, endpoint, duration, original_size,
                compressed_size, num_regiones, num_checkboxes,
                checkboxes_asociados, avg_association_conf, success, error, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now(),
                data.get("filename"),
                data.get("endpoint"),
                data.get("duration"),
                data.get("original_size"),
                data.get("compressed_size"),
                data.get("num_regiones"),
                data.get("num_checkboxes"),
                data.get("checkboxes_asociados"),
                data.get("avg_association_conf"),
                data.get("success", True),
                data.get("error"),
                json.dumps(data.get("metadata", {})),
            ),
        )
        self.conn.commit()
