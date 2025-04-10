#!/usr/bin/env python3

import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import main as rag_engine
from ocr_service import extract_text_from_pdf_bytes

# Asegurarnos de que existe el directorio para las plantillas
os.makedirs("templates", exist_ok=True)

app = FastAPI(title="ISDM Chatbot")

# Configurar templates
templates = Jinja2Templates(directory="templates")

# Modelo para la solicitud de chat, ahora incluye conversation_history
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/pdf-extractor", response_class=HTMLResponse)
async def pdf_extractor(request: Request):
    return templates.TemplateResponse("pdf_extractor.html", {"request": request})

@app.post("/extract-text/")
async def extract_text_endpoint(file: UploadFile = File(...)):
    # Leer el archivo PDF
    pdf_content = await file.read()
    
    # Extraer texto usando la función de OCR
    result = extract_text_from_pdf_bytes(pdf_content)
    
    # Añadir el nombre del archivo a los metadatos
    result["filename"] = file.filename
    
    return result

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    query = chat_request.message
    conv_history = chat_request.conversation_history  # Se obtiene el historial de conversación
    
    if not query:
        return JSONResponse(content={"response": "Por favor, escribe un mensaje."})
    
    try:
        # Inicializar ChromaDB
        _, collection = rag_engine.setup_chroma()
        
        # Aumentar la consulta para obtener mejores resultados
        augmented_queries = rag_engine.augment_query(query)
        
        # Recuperar contexto relevante
        context_docs = rag_engine.retrieve_context_multi(augmented_queries, collection)
        
        # Generar respuesta, incluyendo el historial de conversación
        response = rag_engine.generate_response(query, context_docs, conv_history)
        
        return {"response": response}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    try:
        print("Iniciando la aplicación web del chatbot con FastAPI...")
        uvicorn.run(app, host="0.0.0.0", port=4000)
    except Exception as e:
        print(f"Error al iniciar el servidor: {str(e)}")
        import traceback
        traceback.print_exc()
