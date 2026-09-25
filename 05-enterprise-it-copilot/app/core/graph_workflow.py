import os
from typing import List, TypedDict, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler

from app.core.rag_engine import hybrid_rerank_search

load_dotenv()

# Handler de Langfuse
langfuse_handler = CallbackHandler()

# Definición del Estado
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retry_count: int

# LLM Principal
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# NODOS
def retrieve_node(state: GraphState, config: RunnableConfig) -> dict:
    question = state["question"]
    reranked_docs = hybrid_rerank_search(question, config=config)
    return {
        "documents": reranked_docs,
        "retry_count": state.get("retry_count", 0)
    }

def grade_documents_node(state: GraphState, config: RunnableConfig) -> dict:
    question = state["question"]
    documents = state["documents"]
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un evaluador estricto. Responde únicamente 'si' si el documento contiene información directamente relacionada con la pregunta, o 'no' si no la contiene."),
        ("human", "Pregunta: {question}\nDocumento: {document}")
    ])
    grader_chain = grader_prompt | llm | StrOutputParser()
    
    valid_docs = []
    for doc in documents:
        res = grader_chain.invoke({"question": question, "document": doc.page_content}, config=config).strip().lower()
        if "si" in res:
            valid_docs.append(doc)
            
    return {"documents": valid_docs}

def rewrite_query_node(state: GraphState, config: RunnableConfig) -> dict:
    current_retry = state.get("retry_count", 0) + 1
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un optimizador de consultas técnicas. Reescribe la consulta para hacerla clara y concisa. Responde SOLO con la consulta en texto plano."),
        ("human", "{question}")
    ])
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    new_question = rewrite_chain.invoke({"question": state["question"]}, config=config).strip()
    
    return {
        "question": new_question,
        "retry_count": current_retry
    }

def human_intervention_node(state: GraphState) -> dict:
    return {}

def generate_node(state: GraphState, config: RunnableConfig) -> dict:
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

# BORDES CONDICIONALES
def decide_to_generate(state: GraphState) -> str:
    valid_docs = state.get("documents", [])
    retries = state.get("retry_count", 0)
    
    if valid_docs:
        return "generate"
    elif retries < 1:
        return "rewrite"
    else:
        return "human_intervention"

# COMPILACIÓN DEL GRAFO
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
compiled_graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_intervention"]
)

# FUNCIONES AUXILIARES PARA FASTAPI Y GRADIO
def run_copilot_workflow(question: str, thread_id: str) -> Dict[str, Any]:
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "tags": ["Enterprise-Copilot", "CRAG", "HITL"],
        "metadata": {
            "user_id": "ops_engineer"
        }
    }
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retry_count": 0
    }
    
    for _ in compiled_graph.stream(initial_state, config):
        pass
        
    state_snapshot = compiled_graph.get_state(config)
    
    if state_snapshot.next and "human_intervention" in state_snapshot.next:
        return {
            "status": "PAUSED_HITL",
            "message": "Consulta no resuelta automáticamente. Se requiere revisión humana.",
            "question": question,
            "thread_id": thread_id
        }
        
    final_values = state_snapshot.values
    return {
        "status": "COMPLETED",
        "generation": final_values.get("generation", "No se obtuvo respuesta."),
        "documents": [doc.page_content for doc in final_values.get("documents", [])],
        "thread_id": thread_id
    }

def resume_hitl_workflow(thread_id: str, new_question: Optional[str] = None) -> Dict[str, Any]:
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "tags": ["Enterprise-Copilot", "CRAG", "HITL"],
        "metadata": {
            "user_id": "ops_engineer"
        }
    }
    
    if new_question:
        compiled_graph.update_state(
            config,
            {"question": new_question, "retry_count": 0},
            as_node="human_intervention"
        )
        
    for _ in compiled_graph.stream(None, config):
        pass
        
    final_values = compiled_graph.get_state(config).values
    
    # 📌 Solución al error: verificación segura de flush()
    if hasattr(langfuse_handler, "flush"):
        langfuse_handler.flush()
    elif hasattr(langfuse_handler, "langfuse") and hasattr(langfuse_handler.langfuse, "flush"):
        langfuse_handler.langfuse.flush()
    
    return {
        "status": "COMPLETED",
        "generation": final_values.get("generation", "No se obtuvo respuesta."),
        "documents": [doc.page_content for doc in final_values.get("documents", [])],
        "thread_id": thread_id
    }