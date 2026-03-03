# OCR AIDA Pro

**API avanzada para extracción de texto y datos estructurados de documentos académicos mediante OCR, con soporte para checkboxes, tablas y generación de PDF. (Aun en proceso...)**

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/)
[![Tesseract](https://img.shields.io/badge/Tesseract-5.5.2-orange)](https://github.com/tesseract-ocr/tesseract)
[![Render](https://img.shields.io/badge/deployed%20on-Render-46E3B7)](https://render.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/status-production%20stable-brightgreen)]()

---

## Descripción

OCR AIDA Pro es una API REST desarrollada con FastAPI que permite procesar imágenes de documentos (comunicados, formularios, artículos) para extraer no solo el texto completo, sino también información estructurada como fechas, horarios, materiales y checkboxes. Incorpora un pipeline de preprocesamiento de imágenes, un motor OCR basado en Tesseract, y módulos de post-procesamiento inteligente. Además, ofrece procesamiento asíncrono para imágenes grandes y un endpoint para generar un PDF con la disposición original del texto. (Se sigue trabajando en ello...)

---

## Estado del proyecto

- **Versión actual del OCR:** `10.3.5`  
- **Estado:** Estable, en producción.  
- **Versión actual del generador del PDF:** `1.2`  
- **Estado:** Estable, en producción.  
- **URL de la API desplegada:** [https://api-ocr-g2g4.onrender.com](https://api-ocr-g2g4.onrender.com)  

---

## Características principales

- **OCR robusto** con Tesseract 5.5.2 (binario estático incluido, sin dependencias externas).
- **Preprocesamiento avanzado**: Corrección de inclinación (skew), eliminación de sombras, reducción de ruido, binarización adaptativa (Otsu, Sauvola, Adaptive Threshold).
- **Detección de checkboxes**: identifica cuadrados `[ ]`, círculos `O` e incisos `(a)`; determina si están marcados con análisis de densidad y confianza OCR.
- **Asociación inteligente pregunta-respuesta**: Algoritmo multi-nivel que asocia cada checkbox con el texto más cercano (distancia, alineación, tamaño de fuente y confianza OCR). Soporte para grupos (radio buttons).
- **Extracción estructurada de datos**: Detecta automáticamente horarios (ej. `8:00 AM`), días de la semana, fechas (ej. `24 de febrero de 2026`) y materiales mediante expresiones regulares.
- **Procesamiento asíncrono**: Endpoint `/ocr/async` para imágenes >5 MB con seguimiento mediante `task_id`.
- **Generación de PDF con coordenadas exactas**: endpoint `/ocr/pdf` que recrea la disposición original del texto usando ReportLab y fuente Unicode (DejaVuSans).
- **Segmentación y paralelización**: divide el documento en regiones y las procesa simultáneamente (hasta 5 en paralelo) para optimizar el tiempo de respuesta.
- **Compresión inteligente**: Redimensiona imágenes a 2000 píxeles máximo y ajusta la calidad JPEG para evitar timeouts en Render.
- **Métricas de calidad**: Sistema integrado con SQLite que registra cada solicitud para análisis de rendimiento y depuración.
- **CORS habilitado** Para uso desde cualquier frontend.
- **Manejo robusto de timeouts**: Respuestas 504 cuando Tesseract excede el tiempo límite.

---

## Tecnologías utilizadas

| Categoría | Tecnologías |
|:---|:---|
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **OCR** | Tesseract 5.5.2 (binario estático), pytesseract |
| **Visión por Computador** | OpenCV, scikit-image |
| **Generación de PDF** | ReportLab 4.2.2 |
| **Base de datos** | SQLite (métricas) |
| **Frontend de prueba** | HTML, CSS, JavaScript (InfinityFree) |
| **Despliegue** | Render (producción), InfinityFree (pruebas) |
| **Otras librerías** | numpy, python-multipart, jinja2 (opcional), pyspellchecker (opcional) |

---

## Instalación y configuración local

### Requisitos previos
- Python 3.12 o superior
- Git

### Pasos

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/Cat-Not-Furry/API-OCR.git
   cd API-OCR
   ```
2. **Crear y activar entorno virtual**

   ```bash
   python -m venv venv
   source venv/bin/activate # Linux/Mac

   venv\Scripts\activate # Windows
   ```
3. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```
4. **Estructura de carpetas necesaria (ya incluia en el repositorio)**

   **API-OCR/**  
   **│**  
   **├─ bin/          # Binario estático de Tesseract**  
   **├─ tessdata/     # Archivos traineddata (spa, eng)**  
   **└─ fonts/        # Fuentes para PDF (DejaVuSans.ttf)**
5. **Ejecutar localmente**

   ```bash
   uvicorn main:app --reload
   ```
   **La API estara disponible en http://localhost:8000**  

## Variables de entorno (opcionales)
- **PORT: puerto del servidor (por defecto 10000 en Render)**  
- **INFINITYFREE_URL: endpoint para callback a InfinityFree (solo si se usa)**  

## Uso de la API

| Método | Endpoint | Descripción |
|:---|:---|:---|
| **GET** | **/** | **Health check e información del servicio** |
| **GET** | **/health** | **Endpoint simple para monitoreo** |
| **POST** | **/ocr/basico** | **OCR simple sin preprocesamiento** |
| **POST** | **/ocr/segmentado** | **OCR con segmentación por regiones y paralelización** |
| **POST** | **/ocr/tabla** | **Extracción de texto de tablas detectadas** |
| **POST** | **/ocr/documento_completo** | **Pipeline inteligente: elige el mejor método según el documento. Permite obtener coordenadas (return_coords) y modo unificado.** |
| **POST** | **/ocr/checkboxes** | **Detecta checkboxes y devuelve pares pregunta‑respuesta, con opción de coordenadas** |
| **POST** | **/ocr/pdf** | **Genera un PDF a partir de las coordenadas de palabras (misma disposición que la imagen)** |
| **POST** | **/ocr/async** | **Procesamiento asíncrono para imágenes >5 MB. Devuelve task_id** |
| **GET** | **/ocr/result/{task_id}** | **Consulta el resultado de una tarea asíncrona** |

## Ejemplos de uso:
**cURL (linea de comandos)**

```bash
# OCR básico
curl -X POST https://api-ocr-g2g4.onrender.com/ocr/basico \
  -F "file=@imagen.jpg" \
  -F "lang=spa"    # Puedes usar eng si el documento esta en Ingles

# Documento completo con coordenadas
curl -X POST https://api-ocr-g2g4.onrender.com/ocr/documento_completo \
  -F "file=@imagen.jpg" \
  -F "lang=spa" \    # Puedes usar eng si el documento esta en Ingles
  -F "return_coords=true"

# Nota: pudes agrgar el parametro "-v" si quieres ver los logs 
```
**Python (con request)**

```python
import requests

url = "https://api-ocr-g2g4.onrender.com/ocr/documento_completo"
files = {"file": open("documento.jpg", "rb")}
data = {"lang": "spa", "return_coords": "true"}

response = requests.post(url, files=files, data=data)
resultado = response.json()
print(resultado["texto_completo"])
print(resultado["texto_estructurado"])
```
**JavaScript (para frontend)**

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('lang', 'spa');

fetch('https://api-ocr-g2g4.onrender.com/ocr/documento_completo', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data.texto_completo));
```
## Respuesta típica (endpoint /ocr/documento_completo)

```json
{
  "success": true,
  "filename": "test_complejo.jpg",
  "texto_completo": "COMUNICADO A PADRES DE FAMILIA...",
  "texto_estructurado": {
    "texto_limpio": "COMUNICADO A PADRES DE FAMILIA...",
    "horarios": [],
    "dias": ["martes 24"],
    "materiales": [],
    "notas": "",
    "fechas": ["20 de febrero de 2026", "24 de febrero de 2026"]
  },
  "metadata": {
    "language": "spa",
    "optimizacion": "texto",
    "lineas_detectadas": 247,
    "correct_spelling": false
  },
  "coordenadas": [ ... ]  // solo si return_coords=true
}
```

# Estructura del proyecto

```text
API-OCR/
├── main.py                 # Punto de entrada, endpoints
├── config.py               # Constantes globales (rutas, límites)
├── background.py           # Gestión de tareas asíncronas
├── metrics.py              # Sistema de métricas SQLite
├── requirements.txt        # Dependencias
├── bin/                    # Binario estático de Tesseract
│   └── tesseract
├── tessdata/               # Archivos de idioma
│   ├── spa.traineddata
│   └── eng.traineddata
├── fonts/                  # Fuentes para PDF
│   └── DejaVuSans.ttf
├── utils/                  # Utilidades
│   ├── file_handling.py    # Validación, lectura, compresión
│   └── logging_config.py   # Configuración de logs
├── preprocessing/          # Procesamiento de imágenes
│   ├── enhance.py          # Corrección skew, sombras, ruido
│   ├── detection.py        # Detección de tablas, regiones
│   ├── compression.py      # Redimensionado y compresión
│   └── checkbox.py         # Detección de checkboxes
├── ocr/                    # Motor OCR
│   ├── engine.py           # Ejecución de Tesseract
│   ├── postprocess.py      # Limpieza y extracción estructurada
│   └── association.py      # Asociación checkboxes-texto
└── integration/            # Integración con InfinityFree
    └── infinityfree.py     # Cliente para callback
```
## Despliegue en RENDER

1. **Crear una cuenta en [render.com](render.com)**
2. **Conectar el repositorio (GitHub/GitLab/Bitbucket)**
3. **Crear un nuevo Web Service con los siguientes parámetros:**
 - **Build Command: pip install -r requirements.txt**
 - **Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT**
 - **Python Version: 3.12 (opcional, se puede especificar en variables de entorno)**
4. **Añadir variable de entorno (opcional): PYTHON_VERSION=3.12.8**
5. **Desplegar**

**Nota: El binario estático de Tesseract ya está incluido en el repositorio, por lo que no se requiere instalación adicional en Render.** 
## Metricas
**El proyecto incorpora un sistema de métricas basado en SQLite (ocr_metrics.db). Cada solicitud a los endpoints principales queda registrada con:**

 - **Duración**
 - **Tamaño original y comprimido**
 - **Número de regiones/checkboxes**
 - **Confianza de asociación**
 - **Estado de éxito/error**

**Esto permite monitorizar el rendimiento y detectar patrones de fallo.**

## Contribuciones
**Este proyecto es de uso libre. Si deseas contribuir, por favor abre un issue/pull request en el repositorio.**

# Licencia 
## GNU General Public License v3.0 Copyright (c) 2026 OCR AIDA Pro Team.

# Contacto
## Desarrollador Principal: [Cat-Not-Furry] (https://github.com/Cat-Not-Furry)

# Agradecimientos

## ° A Tesseract OCR por su potente motor de reconocimiento.
## ° A Render por el despliegue gratuito y confiable.
## ° A InfinityFree por el alojamiento de las páginas de prueba.
## ° A ReportLab por la generación de PDF precisa.
## ° A Cursor y DeepSeek por su ayuda en la planificación y desarrollo.

<h1>© 2026 OCR AIDA Pro - Desarrollado con ❤️ para documentos académicos.</h1>
