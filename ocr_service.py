import os
import pytesseract
from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image

# Load OpenAI API key from environment variable (if needed)
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI()

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    """
    Extracts text from a PDF using OCR (via Tesseract) and returns it in Markdown format.
    """
    # Convert PDF to images
    images = convert_from_bytes(await file.read())

    extracted_text = []
    
    for i, img in enumerate(images):
        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(img)  # Extract text from image using Tesseract

        # Format the text for markdown
        page_title = f"## Page {i + 1}\n"  # Markdown header for each page
        page_text = f"{text}\n\n"  # Add some space after the page text

        extracted_text.append(page_title + page_text)  # Append page title and content

    # Join all pages' content into a single Markdown-formatted string
    markdown_content = "\n".join(extracted_text)

    return {"markdown_text": markdown_content}
