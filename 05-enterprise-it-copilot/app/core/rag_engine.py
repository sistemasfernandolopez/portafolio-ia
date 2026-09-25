import os
from typing import List
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

load_dotenv()

# Documentos iniciales de la base de conocimiento
documents_raw = [
    "La arquitectura de microservicios utiliza gRPC para comunicación interna de alta velocidad y baja latencia.",
    "El código de error ERR-902 indica un fallo crítico de timeout en la conexión con la base de datos PostgreSQL.",
    "Para reducir el estrés y la migraña, se recomienda hacer pausas activas, hidratación constante y ejercicios de respiración.",
    "El sistema de autenticación OAuth2 requiere un token JWT firmado con el algoritmo RS256.",
    "En caso de ver el error ERR-902 en los logs, reinicie el servicio de caché Valkey antes de escalar a soporte."
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.create_documents(documents_raw)

# Inicializar recuperadores
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vectorstore = FAISS.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

compressor = CohereRerank(
    model="rerank-multilingual-v3.0",
    top_n=2,
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

def hybrid_rerank_search(query: str, config: RunnableConfig = None) -> List[Document]:
    """Combina BM25 + FAISS y aplica Re-ranking con Cohere."""
    bm25_docs = bm25_retriever.invoke(query, config=config)
    vector_docs = vector_retriever.invoke(query, config=config)
    
    seen = set()
    combined_docs = []
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined_docs.append(doc)
            
    if not combined_docs:
        return []
        
    return compressor.compress_documents(documents=combined_docs, query=query)

def add_documents_to_knowledge_base(new_texts: List[str]):
    """Permite añadir nuevo contenido a la base de conocimiento en caliente."""
    global bm25_retriever, vectorstore, vector_retriever
    
    new_docs = text_splitter.create_documents(new_texts)
    docs.extend(new_docs)
    
    # Reconstruir retrievers
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4
    vectorstore.add_documents(new_docs)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return len(new_docs)