import os
import sys
import logging
from typing import List, TypedDict

# ------------------------------------------------------------------
# 0. CONFIGURACIÓN DE REGISTROS Y ADVERTENCIAS
# ------------------------------------------------------------------
LOG_FILE = "app.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.captureWarnings(True)
sys.stderr = open(LOG_FILE, "a", encoding="utf-8")

from dotenv import load_dotenv
from operator import itemgetter

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Falta la variable GOOGLE_API_KEY en el archivo .env")

if not os.getenv("COHERE_API_KEY"):
    raise ValueError("Falta la variable COHERE_API_KEY en el archivo .env")

# LangChain & LangGraph imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --------------------------------------------------
# 1. DOCUMENTOS RAW Y PIPELINE DE BÚSQUEDA HÍBRIDA
# --------------------------------------------------
documents_raw = [
    "La arquitectura de microservicios utiliza gRPC para comunicación interna de alta velocidad y baja latencia.",
    "El código de error ERR-902 indica un fallo crítico de timeout en la conexión con la base de datos PostgreSQL.",
    "Para reducir el estrés y la migraña, se recomienda hacer pausas activas, hidratación constante y ejercicios de respiración.",
    "El sistema de autenticación OAuth2 requiere un token JWT firmado con el algoritmo RS256.",
    "En caso de ver el error ERR-902 en los logs, reinicie el servicio de caché Valkey antes de escalar a soporte."
]

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.create_documents(documents_raw)

# Retrievers: Léxico (BM25) + Vectorial (FAISS)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vectorstore = FAISS.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Re-ranker con Cohere
compressor = CohereRerank(
    model="rerank-multilingual-v3.0",
    top_n=2,
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

def hybrid_rerank_search(query: str) -> List[Document]:
    """Combina BM25 + FAISS y aplica Re-ranking con Cohere."""
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    
    seen = set()
    combined_docs = []
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined_docs.append(doc)
            
    if not combined_docs:
        return []
        
    return compressor.compress_documents(documents=combined_docs, query=query)

# ------------------------------------------------------------------
# 2. ESTADO DEL GRAFO (GraphState)
# ------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int

# ------------------------------------------------------------------
# 3. COMPONENTES LLM
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ------------------------------------------------------------------
# 4. NODOS DEL GRAFO
# ------------------------------------------------------------------
def retrieve_node(state: GraphState) -> dict:
    """Nodo 1: Ejecuta la búsqueda híbrida + Re-ranking."""
    question = state["question"]
    print(f"\n🔎 [Nodo Retrieve] Buscando información para: '{question}'...")
    
    reranked_docs = hybrid_rerank_search(question)
    return {
        "documents": reranked_docs,
        "retry_count": state.get("retry_count", 0)
    }

def grade_documents_node(state: GraphState) -> dict:
    """Nodo 2: Valida si los documentos son realmente útiles para responder."""
    print("⚖️ [Nodo Grade] Evaluando relevancia del contexto recuperado...")
    question = state["question"]
    documents = state["documents"]
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un evaluador estricto. Responde únicamente 'si' si el documento contiene información directamente relacionada con la pregunta, o 'no' si no la contiene."),
        ("human", "Pregunta: {question}\nDocumento: {document}")
    ])
    grader_chain = grader_prompt | llm | StrOutputParser()
    
    valid_docs = []
    for doc in documents:
        res = grader_chain.invoke({"question": question, "document": doc.page_content}).strip().lower()
        if "si" in res:
            valid_docs.append(doc)
            
    print(f"   └─ Documentos válidos tras filtrado: {len(valid_docs)} de {len(documents)}")
    return {"documents": valid_docs}

def rewrite_query_node(state: GraphState) -> dict:
    current_retry = state.get("retry_count", 0) + 1
    print(f"🔄 [Nodo Rewrite] Reintentando consulta por automatización (Intento {current_retry}/1)...")
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "Eres un optimizador de consultas de búsqueda técnica. "
            "Tu única tarea es reescribir la consulta dada para hacerla más concisa y clara.\n\n"
            "REGLA CRÍTICA: Devuelve ÚNICAMENTE la frase reescrita en texto plano. "
            "NO incluyas introducciones, NO uses formato Markdown, NO des consejos ni explicaciones."
        ),
        ("human", "{question}")
    ])
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    new_question = rewrite_chain.invoke({"question": state["question"]}).strip()
    print(f"   └─ Nueva consulta reescrita: '{new_question}'")
    
    return {
        "question": new_question,
        "retry_count": current_retry
    }

def human_intervention_node(state: GraphState) -> dict:
    """Nodo 4: Punto de pausa para intervención humana (HITL)."""
    print("\n🛑 [Nodo Human Intervention] El sistema automático no encontró respuesta relevante tras 1 reintento.")
    print("   Espereando feedback o actualización manual del operador...")
    return {}

def generate_node(state: GraphState) -> dict:
    """Nodo 5: Genera la respuesta final usando el contexto validado."""
    print("🤖 [Nodo Generate] Generando la respuesta final...")
    question = state["question"]
    documents = state.get("documents", [])
    
    context = "\n\n".join(doc.page_content for doc in documents) if documents else "No hay contexto disponible."
    
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente técnico especializado. Responde a la pregunta basándote únicamente en el contexto proporcionado:\n\n{context}"),
        ("human", "{question}")
    ])
    gen_chain = gen_prompt | llm | StrOutputParser()
    
    generation = gen_chain.invoke({"context": context, "question": question})
    return {"generation": generation}

# ------------------------------------------------------------------
# 5. BORDES CONDICIONALES
# ------------------------------------------------------------------
def decide_to_generate(state: GraphState) -> str:
    """Evalúa los resultados del filtrado y decide el siguiente nodo."""
    valid_docs = state.get("documents", [])
    retries = state.get("retry_count", 0)
    
    if valid_docs:
        return "generate"
    elif retries < 1:
        return "rewrite"
    else:
        # Si ya gastó el único reintento permitido, deriva a intervención humana
        return "human_intervention"

# ------------------------------------------------------------------
# 6. COMPILACIÓN DEL GRAFO CON CHECKPOINTER (HITL)
# ------------------------------------------------------------------
workflow = StateGraph(GraphState)

# Registrar nodos
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("human_intervention", human_intervention_node)
workflow.add_node("generate", generate_node)

# Registrar bordes
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite_query",
        "human_intervention": "human_intervention"
    }
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("human_intervention", "retrieve")
workflow.add_edge("generate", END)

# Checkpointer en memoria para persistir el estado e interrupción
memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_intervention"]  # Se detiene justo antes de este nodo
)

# ------------------------------------------------------------------
# 7. EJECUCIÓN CON PRUEBA DE HUMAN-IN-THE-LOOP
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Probamos con una consulta fuera de dominio para forzar el flujo completo:
    # Retrieve -> Grade (Falla) -> Rewrite (Intento 1) -> Retrieve -> Grade (Falla) -> HITL
    query_ambigua = "¿Cómo encuentro los datos?"
    
    config = {"configurable": {"thread_id": "sesion_demo_01"}}
    initial_state = {
        "question": query_ambigua,
        "documents": [],
        "generation": "",
        "retry_count": 0
    }
    
    print("========================================================================")
    print("🚀 EJECUTANDO FLUJO DE LANGGRAPH CON 1 REINTENTO Y HUMAN-IN-THE-LOOP")
    print("========================================================================")
    
    # 1. Primera ejecución hasta que se complete o se interrumpa
    for event in app.stream(initial_state, config):
        pass

    # 2. Comprobar si el grafo fue pausado por la interrupción programada
    state_snapshot = app.get_state(config)
    
    if state_snapshot.next and "human_intervention" in state_snapshot.next:
        print("\n" + "="*60)
        print("👤 INTERVENCIÓN HUMANA REQUERIDA")
        print("="*60)
        print("El sistema automático no pudo encontrar contexto relevante tras 1 reintento.")
        
        # Entrada por consola para simular la acción del operador humano
        nueva_query_humana = input("\n👉 Ingresa una consulta más específica (ej. 'solución error ERR-902') o presiona ENTER para forzar respuesta: ")
        
        if nueva_query_humana.strip():
            # El humano actualiza la pregunta y reinicia los reintentos
            app.update_state(
                config, 
                {"question": nueva_query_humana, "retry_count": 0}, 
                as_node="human_intervention"
            )
            print(f"✅ Consulta corregida por el humano a: '{nueva_query_humana}'")
        else:
            print("⚠️ El humano decidió no modificar la consulta. Se procederá con la información disponible.")
            
        print("\n▶️ Reanudando ejecución del grafo tras la intervención...")
        
        # Reanudar la ejecución desde la pausa
        for event in app.stream(None, config):
            pass

    # 3. Mostrar estado final
    final_state = app.get_state(config).values
    print("\n================ RESULTADO FINAL ================")
    print(final_state.get("generation", "No se generó respuesta."))