from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.core.graph_workflow import run_copilot_workflow, resume_hitl_workflow
from app.core.crew_agents import run_tech_news_crew
from app.core.rag_engine import add_documents_to_knowledge_base

router = APIRouter(prefix="/api/v1")

class ChatRequest(BaseModel):
    question: str
    thread_id: str

class HITLResolveRequest(BaseModel):
    thread_id: str
    new_question: Optional[str] = None

class ResearchRequest(BaseModel):
    topic: str

class IngestRequest(BaseModel):
    texts: List[str]

@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        return run_copilot_workflow(question=req.question, thread_id=req.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hitl/resolve")
def resolve_hitl_endpoint(req: HITLResolveRequest):
    try:
        return resume_hitl_workflow(thread_id=req.thread_id, new_question=req.new_question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/research")
def research_endpoint(req: ResearchRequest):
    try:
        report = run_tech_news_crew(topic=req.topic)
        return {"status": "SUCCESS", "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/add")
def ingest_endpoint(req: IngestRequest):
    try:
        added_count = add_documents_to_knowledge_base(req.texts)
        return {"status": "SUCCESS", "added_chunks": added_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))