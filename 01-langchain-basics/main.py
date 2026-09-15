import os
import sys
import logging
import warnings

# ------------------------------------------------------------------
# 0. CONFIGURACIÓN DE REGISTROS Y ADVERTENCIAS
# ------------------------------------------------------------------
LOG_FILE = "app.log"

# 1. Configurar el logger principal
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 2. Capturar todas las advertencias nativas de Python (incluido DeprecationWarning)
logging.captureWarnings(True)

# 3. Redirigir stderr globalmente a app.log antes de realizar importaciones
sys.stderr = open(LOG_FILE, "a", encoding="utf-8")

from dotenv import load_dotenv
from operator import itemgetter

# Cargar variables de entorno
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Falta la variable GOOGLE_API_KEY en el archivo .env")

if not os.getenv("COHERE_API_KEY"):
    raise ValueError("Falta la variable COHERE_API_KEY en el archivo .env")

# Importaciones de LangChain (cualquier aviso aquí irá directamente a app.log)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ------------------------------------------------------------------
# 1. DOCUMENTOS DE EJEMPLO
# ------------------------------------------------------------------
documents_raw = [
    "La arquitectura de microservicios utiliza gRPC para comunicación interna de alta velocidad y baja latencia.",
    "El código de error ERR-902 indica un fallo crítico de timeout en la conexión con la base de datos PostgreSQL.",
    "Para reducir el estrés y la migraña, se recomienda hacer pausas activas, hidratación constante y ejercicios de respiración.",
    "El sistema de autenticación OAuth2 requiere un token JWT firmado con el algoritmo RS256.",
    "En caso de ver el error ERR-902 en los logs, reinicie el servicio de caché Valkey antes de escalar a soporte."
]

# ------------------------------------------------------------------
# 2. CHUNKING
# ------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.create_documents(documents_raw)

# ------------------------------------------------------------------
# 3. RETRIEVERS (LÉXICO + VECTORIAL)
# ------------------------------------------------------------------
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vectorstore = FAISS.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def hybrid_search(query: str):
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    
    seen = set()
    combined_docs = []
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined_docs.append(doc)
    return combined_docs

# ------------------------------------------------------------------
# 4. RE-RANKING CON COHERE
# ------------------------------------------------------------------
compressor = CohereRerank(
    model="rerank-multilingual-v3.0", 
    top_n=2,
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

def get_reranked_docs(query: str):
    docs = hybrid_search(query)
    return compressor.compress_documents(documents=docs, query=query)

# ------------------------------------------------------------------
# 5. LLM Y PIPELINE LCEL
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente técnico especializado. Responde únicamente basándote en el siguiente contexto:\n\n{context}"),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": itemgetter("input") | RunnableLambda(get_reranked_docs) | format_docs, 
        "input": itemgetter("input")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------------------------------------------------------
# 6. EJECUCIÓN
# ------------------------------------------------------------------
if __name__ == "__main__":
    query = "¿Qué debo hacer si aparece el error ERR-902?"
    
    print(f"🔎 Consulta: {query}\n")
    
    retrieved_docs = get_reranked_docs(query)

    print("--- Fragmentos seleccionados tras Búsqueda Híbrida + Cohere Re-ranking ---")
    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"[{idx}] {doc.page_content}")
    print("------------------------------------------------------------------------\n")
    
    response = rag_chain.invoke({"input": query})

    print("🤖 Respuesta final de Gemini:")
    print(response)