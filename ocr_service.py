import os
import pytesseract
from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
import io
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ocr_service")

# Create FastAPI app for standalone service
app = FastAPI()

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extrae texto de un PDF usando OCR (vía Tesseract) y lo devuelve en formato Markdown.
    Esta función puede ser importada y usada desde otros módulos.
    
    Args:
        pdf_bytes (bytes): Contenido del PDF en formato bytes
    
    Returns:
        dict: Diccionario con el texto extraído en formato Markdown y metadatos
    """
    try:
        # Convertir PDF a imágenes
        logger.info("Convirtiendo PDF a imágenes...")
        images = convert_from_bytes(pdf_bytes)
        
        extracted_text = []
        page_count = len(images)
        
        logger.info(f"Procesando {page_count} páginas con OCR...")
        for i, img in enumerate(images):
            # Realizar OCR usando Tesseract
            logger.info(f"Procesando página {i+1}/{page_count}")
            text = pytesseract.image_to_string(img, lang='spa')  # Extraer texto de la imagen usando Tesseract
            
            # Formatear el texto para markdown
            page_title = f"## Página {i + 1}\n"  # Encabezado markdown para cada página
            page_text = f"{text}\n\n"  # Añadir espacio después del texto de la página
            
            extracted_text.append(page_title + page_text)  # Unir el título de la página y el contenido
        
        # Unir el contenido de todas las páginas en una cadena formateada en Markdown
        markdown_content = "\n".join(extracted_text)
        
        return {
            "markdown_text": markdown_content,
            "page_count": page_count,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error al extraer texto del PDF: {str(e)}")
        return {
            "markdown_text": "",
            "page_count": 0,
            "status": "error",
            "error": str(e)
        }

def extract_text_from_pdf_path(pdf_path):
    """
    Extrae texto de un PDF usando OCR (vía Tesseract) a partir de una ruta de archivo.
    
    Args:
        pdf_path (str): Ruta al archivo PDF
    
    Returns:
        dict: Diccionario con el texto extraído en formato Markdown y metadatos
    """
    try:
        # Convertir PDF a imágenes
        logger.info(f"Convirtiendo PDF {pdf_path} a imágenes...")
        images = convert_from_path(pdf_path)
        
        extracted_text = []
        page_count = len(images)
        
        logger.info(f"Procesando {page_count} páginas con OCR...")
        for i, img in enumerate(images):
            # Realizar OCR usando Tesseract
            logger.info(f"Procesando página {i+1}/{page_count}")
            text = pytesseract.image_to_string(img, lang='spa')  # Extraer texto de la imagen usando Tesseract
            
            # Formatear el texto para markdown
            page_title = f"## Página {i + 1}\n"  # Encabezado markdown para cada página
            page_text = f"{text}\n\n"  # Añadir espacio después del texto de la página
            
            extracted_text.append(page_title + page_text)  # Unir el título de la página y el contenido
        
        # Unir el contenido de todas las páginas en una cadena formateada en Markdown
        markdown_content = "\n".join(extracted_text)
        
        return {
            "markdown_text": markdown_content,
            "page_count": page_count,
            "status": "success",
            "source": os.path.basename(pdf_path)
        }
    except Exception as e:
        logger.error(f"Error al extraer texto del PDF: {str(e)}")
        return {
            "markdown_text": "",
            "page_count": 0,
            "status": "error",
            "error": str(e),
            "source": os.path.basename(pdf_path)
        }

@app.post("/extract-text/")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """
    Punto de entrada FastAPI para extraer texto de un PDF usando OCR.
    """
    # Leer el archivo PDF
    pdf_content = await file.read()
    
    # Extraer texto usando la función compartida
    result = extract_text_from_pdf_bytes(pdf_content)
    
    # Añadir el nombre del archivo a los metadatos
    result["filename"] = file.filename
    
    return result
