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

load_dotenv()

# Validaciones de variables de entorno
required_vars = ["GOOGLE_API_KEY", "COHERE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Falta la variable {var} en el archivo .env")

# LangChain, LangGraph & Langfuse imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- IMPORTANTE: Handler de Langfuse ---
from langfuse.callback import CallbackHandler

# ------------------------------------------------------------------
# 1. INICIALIZACIÓN DE LANGFUSE CALLBACK HANDLER
# ------------------------------------------------------------------
langfuse_handler = CallbackHandler(
    tags=["LangGraph", "Corrective-RAG", "HITL", "Gemini-3.5"],
    user_id="developer_demo",
    session_id="sesion_crag_hitl_01"
)

# ------------------------------------------------------------------
# 2. DOCUMENTOS RAW Y PIPELINE DE BÚSQUEDA HÍBRIDA
# ------------------------------------------------------------------
documents_raw = [
    "La arquitectura de microservicios utiliza gRPC para comunicación interna de alta velocidad y baja latencia.",
    "El código de error ERR-902 indica un fallo crítico de timeout en la conexión con la base de datos PostgreSQL.",
    "Para reducir el estrés y la migraña, se recomienda hacer pausas activas, hidratación constante y ejercicios de respiración.",
    "El sistema de autenticación OAuth2 requiere un token JWT firmado con el algoritmo RS256.",
    "En caso de ver el error ERR-902 en los logs, reinicie el servicio de caché Valkey antes de escalar a soporte."
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.create_documents(documents_raw)

# Retrievers
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
    """Combina BM25 + FAISS y aplica Re-ranking registrando en Langfuse."""
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

# ------------------------------------------------------------------
# 3. ESTADO DEL GRAFO (GraphState)
# ------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int

# ------------------------------------------------------------------
# 4. COMPONENTES LLM
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ------------------------------------------------------------------
# 5. NODOS DEL GRAFO (Inyectando RunnableConfig para trazabilidad)
# ------------------------------------------------------------------
def retrieve_node(state: GraphState, config: RunnableConfig) -> dict:
    """Nodo 1: Búsqueda híbrida + Re-ranking."""
    question = state["question"]
    print(f"\n🔎 [Nodo Retrieve] Buscando información para: '{question}'...")
    
    reranked_docs = hybrid_rerank_search(question, config=config)
    return {
        "documents": reranked_docs,
        "retry_count": state.get("retry_count", 0)
    }

def grade_documents_node(state: GraphState, config: RunnableConfig) -> dict:
    """Nodo 2: Valida la relevancia con el LLM."""
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
        # Se pasa `config` para que Langfuse capture cada evaluación individual
        res = grader_chain.invoke({"question": question, "document": doc.page_content}, config=config).strip().lower()
        if "si" in res:
            valid_docs.append(doc)
            
    print(f"   └─ Documentos válidos tras filtrado: {len(valid_docs)} de {len(documents)}")
    return {"documents": valid_docs}

def rewrite_query_node(state: GraphState, config: RunnableConfig) -> dict:
    """Nodo 3: Reescritura de consulta."""
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
    
    new_question = rewrite_chain.invoke({"question": state["question"]}, config=config).strip()
    print(f"   └─ Nueva consulta reescrita: '{new_question}'")
    
    return {
        "question": new_question,
        "retry_count": current_retry
    }

def human_intervention_node(state: GraphState) -> dict:
    """Nodo 4: Punto de pausa para intervención humana (HITL)."""
    print("\n🛑 [Nodo Human Intervention] Requiere retroalimentación del operador...")
    return {}

def generate_node(state: GraphState, config: RunnableConfig) -> dict:
    """Nodo 5: Genera la respuesta final."""
    print("🤖 [Nodo Generate] Generando la respuesta final...")
    question = state["question"]
    documents = state.get("documents", [])
    
    context = "\n\n".join(doc.page_content for doc in documents) if documents else "No hay contexto disponible."
    
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente técnico especializado. Responde a la pregunta basándote únicamente en el contexto proporcionado:\n\n{context}"),
        ("human", "{question}")
    ])
    gen_chain = gen_prompt | llm | StrOutputParser()
    
    generation = gen_chain.invoke({"context": context, "question": question}, config=config)
    return {"generation": generation}

# ------------------------------------------------------------------
# 6. BORDES CONDICIONALES Y COMPILACIÓN
# ------------------------------------------------------------------
def decide_to_generate(state: GraphState) -> str:
    valid_docs = state.get("documents", [])
    retries = state.get("retry_count", 0)
    
    if valid_docs:
        return "generate"
    elif retries < 1:
        return "rewrite"
    else:
        return "human_intervention"

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("human_intervention", human_intervention_node)
workflow.add_node("generate", generate_node)

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

memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_intervention"]
)

# ------------------------------------------------------------------
# 7. EJECUCIÓN CON OBSERVABILIDAD EN LANGFUSE
# ------------------------------------------------------------------
if __name__ == "__main__":
    query_ambigua = "¿Cómo encuentro los datos?"
    
    # 📌 PASO CLAVE: Añadimos `langfuse_handler` en la configuración general
    config = {
        "configurable": {"thread_id": "sesion_demo_01"},
        "callbacks": [langfuse_handler]
    }
    
    initial_state = {
        "question": query_ambigua,
        "documents": [],
        "generation": "",
        "retry_count": 0
    }
    
    print("========================================================================")
    print("🚀 EJECUTANDO FLUJO CON LANGGRAPH + OBSERVABILIDAD LANGFUSE")
    print("========================================================================")
    
    # 1. Primera ejecución hasta la pausa
    for event in app.stream(initial_state, config):
        pass

    state_snapshot = app.get_state(config)
    
    if state_snapshot.next and "human_intervention" in state_snapshot.next:
        print("\n" + "="*60)
        print("👤 INTERVENCIÓN HUMANA REQUERIDA")
        print("="*60)
        
        nueva_query_humana = input("\n👉 Ingresa una consulta más específica (ej. 'solución error ERR-902'): ")
        
        if nueva_query_humana.strip():
            app.update_state(
                config, 
                {"question": nueva_query_humana, "retry_count": 0}, 
                as_node="human_intervention"
            )
            print(f"✅ Consulta corregida a: '{nueva_query_humana}'")
        else:
            print("⚠️ Continuamos sin modificar consulta.")
            
        print("\n▶️ Reanudando ejecución...")
        
        # 2. Continuar ejecución pasando el mismo config con el handler de Langfuse
        for event in app.stream(None, config):
            pass

    final_state = app.get_state(config).values
    print("\n================ RESULTADO FINAL ================")
    print(final_state.get("generation", "No se generó respuesta."))

    # 📌 Asegura que todas las trazas pendientes se envíen a Langfuse Cloud antes de salir
    langfuse_handler.flush()
    print("\n📊 [Langfuse] Trazas enviadas con éxito al Dashboard.")