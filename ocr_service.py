import os
import openai
from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import io

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI()

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    """
    Extracts text from a PDF using OCR (via Tesseract) and OpenAI GPT.
    """
    # Convert PDF to images
    images = convert_from_bytes(await file.read())

    extracted_text = []
    for img in images:
        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(img)  # Extract text from image using Tesseract
        extracted_text.append(text)

    return {"text": "\n".join(extracted_text)}
