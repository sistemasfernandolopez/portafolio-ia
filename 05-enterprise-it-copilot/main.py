import sys
import logging
import uvicorn
from fastapi import FastAPI
import gradio as gr

# Configuración de registros
LOG_FILE = "app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.captureWarnings(True)
sys.stderr = open(LOG_FILE, "a", encoding="utf-8")

from app.api import router as api_router
from app.ui import build_ui

# 1. Crear aplicación FastAPI
app = FastAPI(
    title="Enterprise IT Support & Knowledge Operations Center",
    version="1.0.0",
    description="Copiloto de Soporte IT impulsado por LangChain, LangGraph, Langfuse y CrewAI."
)

# 2. Registrar Router de la API
app.include_router(api_router)

# 3. Construir e integrar la UI de Gradio
gradio_ui = build_ui()
app = gr.mount_gradio_app(app, gradio_ui, path="/")

if __name__ == "__main__":
    print("🚀 Arrancando Servidor Unificado FastAPI + Gradio UI...")
    print("📍 Interfaz Web disponible en: http://localhost:8000")
    print("📍 Documentación Swagger en: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)