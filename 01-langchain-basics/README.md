# 01. RAG Avanzado: Búsqueda Híbrida y Re-Ranking Multi-Modelo 🚀

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-121011?style=flat)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.6_Flash-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![Cohere](https://img.shields.io/badge/Cohere-Rerank_v3-395144?style=flat)](https://cohere.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-00599C?style=flat)](https://github.com/facebookresearch/faiss)

Este proyecto implementa un pipeline de **Recuperación Aumentada por Generación (RAG)** de arquitectura avanzada. Combina búsqueda léxica (BM25) y vectorial (FAISS) mediante un esquema híbrido, optimizado posteriormente con un algoritmo de **Re-Ranking de Cohere** y resuelto con el LLM **Gemini 3.6 Flash** utilizando **LangChain Expression Language (LCEL)**.

---

## 📐 Arquitectura del Sistema

```text
                               ┌──────────────────────────┐
                               │     Documentos Raw       │
                               └────────────┬─────────────┘
                                            │ Chunking (RecursiveCharacterTextSplitter)
                                            ▼
                               ┌──────────────────────────┐
                               │   Documentos Fragmentados│
                               └────────────┬─────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌──────────────────────────┐                  ┌───────────────────────────┐
       │   BM25 Retriever (k=4)   │                  │  FAISS + Google Embeddings│
       │    (Búsqueda Léxica)     │                  │   (Búsqueda Vectorial)    │
       └─────────────┬────────────┘                  └─────────────┬─────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │    Búsqueda Híbrida      │
                               │     (Deduplicación)      │
                               └────────────┬─────────────┘
                                            │ Top Candidates
                                            ▼
                               ┌──────────────────────────┐
                               │  Cohere Re-Ranker v3.0   │
                               │   (Selección Top N=2)    │
                               └────────────┬─────────────┘
                                            │ Contexto Refinado
                                            ▼
                               ┌──────────────────────────┐
                               │    LCEL Pipeline + LLM   │
                               │ (Gemini 3.6 Flash Temp=0)│
                               └────────────┬─────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │     Respuesta Final      │
                               └────────────┬─────────────┘
```

---

## ✨ Características Clave

* **Búsqueda Híbrida (Hybrid Retrieval):** Combina la precisión por palabras clave de **BM25** (ideal para códigos de error exactos como `ERR-902`) con la comprensión semántica de vectores en **FAISS** (`models/gemini-embedding-001`).
* **Re-Ranking Multilingüe (Cohere Rerank v3.0):** Reorganiza los candidatos recuperados evaluando la relevancia cruzada (*cross-encoder*) entre la pregunta y cada fragmento, seleccionando únicamente los $N$ fragmentos más relevantes.
* **Control Anti-Alucinaciones:** Configuración estricta del prompt del sistema para forzar al modelo a responder exclusivamente con base en el contexto recuperado.
* **Pipeline Declarativo LCEL:** Orquestación limpia y modular utilizando LangChain Expression Language y `RunnableLambda`.
* **Manejo Empresarial de Logs:** Captura automática de advertencias (`DeprecationWarning`), `stderr` y logs del sistema redirigidos a `app.log`.

---

## 📁 Estructura del Proyecto

```text
01-langchain-basics/
├── main.py              # Script principal con la cadena RAG completa
├── requirements.txt     # Dependencias fijadas del entorno virtual
├── README.md            # Documentación del proyecto
└── app.log              # Registro de logs y capturas de stderr (generado al ejecutar)
```

---

## 🛠️ Requisitos Previos e Instalación

### 1. Clona el repositorio e ingresa al directorio
```bash
git clone https://github.com/sistemasfernandolopez/portafolio-ia.git
cd portafolio-ia/01-langchain-basics
```

### 2. Crea y activa un entorno virtual
```bash
python -m venv venv
# En Linux/macOS:
source venv/bin/activate
# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 3. Instala las dependencias
```bash
pip install -r requirements.txt
```

### 4. Configura las Variables de Entorno
Crea un archivo `.env` en la raíz de la carpeta con tus claves de API:

```env
GOOGLE_API_KEY=tu_google_api_key
COHERE_API_KEY=tu_cohere_api_key
```

---

## 🚀 Ejecución del Proyecto

Ejecuta el script principal:

```bash
python3 main.py
```

---

## 🧪 Ejemplo de Salida en Consola

```text
🔎 Consulta: ¿Qué debo hacer si aparece el error ERR-902?

--- Fragmentos seleccionados tras Búsqueda Híbrida + Cohere Re-ranking ---
[1] En caso de ver el error ERR-902 en los logs, reinicie el servicio de caché Valkey antes de escalar a soporte.
[2] El código de error ERR-902 indica un fallo crítico de timeout en la conexión con la base de datos PostgreSQL.
------------------------------------------------------------------------

🤖 Respuesta final de Gemini:
Si aparece el error ERR-902, debes reiniciar el servicio de caché Valkey antes de escalar el problema a soporte técnico.
```

---

## 👨‍💻 Conceptos Técnicos Demostrados

| Concepto | Implementación en Código |
| :--- | :--- |
| **Chunking Estratégico** | `RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)` |
| **Vector Store** | `FAISS.from_documents` con `GoogleGenerativeAIEmbeddings` |
| **Sparse Retrieval** | `BM25Retriever.from_documents` |
| **Re-Ranking** | `CohereRerank(model="rerank-multilingual-v3.0", top_n=2)` |
| **Orquestación LCEL** | Uso del operador pipe y `RunnableLambda` para el flujo de datos |
| **Logging & Redirección** | Captura de `sys.stderr` y `logging.captureWarnings(True)` |
