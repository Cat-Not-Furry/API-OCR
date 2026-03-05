FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar solo dependencias del sistema necesarias.
# Tesseract se provee como binario estatico en el repositorio.
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1-mesa-glx \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Respeta PORT de Render y usa 10000 como fallback.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
