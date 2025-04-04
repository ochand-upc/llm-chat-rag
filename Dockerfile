FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-spa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY main.py .
COPY web_app.py .
COPY ocr_service.py .

# Crear directorio para templates
RUN mkdir -p templates

# Copiar archivo de template
COPY templates/index.html templates/

# Crear directorio para datos si no existe
RUN mkdir -p data

# Crear un archivo de acrónimos vacío si no existe
RUN mkdir -p data && echo '{}' > data/acronyms.json

# Crear directorio para la base de datos ChromaDB
RUN mkdir -p chroma_db

# Puerto para la aplicación web
EXPOSE 4000

# Comando por defecto
CMD ["python", "web_app.py"]
