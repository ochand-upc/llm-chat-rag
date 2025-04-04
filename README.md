# LLM Chat RAG con OCR

Este proyecto implementa un sistema de chat con capacidades RAG (Retrieval Augmented Generation) y OCR (Optical Character Recognition) utilizando Docker Compose.

## Componentes

- **RAG Service**: Servicio principal que maneja las consultas, búsqueda de contexto y generación de respuestas con LLM.
- **OCR Service**: Servicio para extraer texto de documentos PDF mediante reconocimiento óptico de caracteres.
- **Web App**: Interfaz web para interactuar con el chatbot.

## Requisitos

- Docker y Docker Compose
- Clave API de OpenAI

## Configuración

1. Copia el archivo de ejemplo `.env.example` a `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edita el archivo `.env` y añade tu clave API de OpenAI:
   ```
   OPENAI_API_KEY=tu_clave_api_aquí
   ```

3. Asegúrate de tener un directorio `data` con el archivo `acronyms.json` para la expansión de acrónimos (se crea automáticamente si no existe).

## Uso

1. Inicia los servicios con Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. Accede a la aplicación web:
   - Web App (Chatbot): http://localhost:4000
   - Servicio OCR: http://localhost:8000/docs

3. Para detener los servicios:
   ```bash
   docker-compose down
   ```

## Estructura de directorios

```
.
├── data/                 # Directorio para archivos de datos (acrónimos, documentos)
│   └── acronyms.json     # Diccionario de acrónimos
├── chroma_db/            # Base de datos vectorial ChromaDB (se crea automáticamente)
├── templates/            # Plantillas HTML para la web app
│   └── index.html        # Interfaz de usuario del chatbot
├── main.py               # Motor RAG principal
├── ocr_service.py        # Servicio OCR
├── web_app.py            # Aplicación web FastAPI
├── docker-compose.yml    # Configuración de Docker Compose
├── Dockerfile            # Dockerfile para el servicio RAG
├── Dockerfile.ocr        # Dockerfile para el servicio OCR
├── requirements.txt      # Dependencias de Python
└── .env                  # Variables de entorno (no incluido en el repositorio)
```

## Endpoints API

### Web App
- `GET /`: Interfaz web del chatbot
- `POST /chat`: Endpoint para enviar mensajes al chatbot

### OCR Service
- `POST /extract-text/`: Endpoint para extraer texto de archivos PDF

## Personalización

- Modifica `acronyms.json` para añadir más acrónimos específicos de tu dominio.
- Ajusta las variables de entorno en `.env` según sea necesario.

## Notas

- El servicio OCR utiliza Tesseract con soporte para español (`spa`). Si necesitas otros idiomas, modifica los Dockerfiles.
- Las respuestas del chatbot incluyen citas a las fuentes utilizadas para generar la respuesta.
