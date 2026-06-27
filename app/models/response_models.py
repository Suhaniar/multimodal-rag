from pydantic import BaseModel
from typing import Any, List, Dict, Optional

class RetrievedChunk(BaseModel):
    chunk: str
    score: float
    source: str
    modality: str
    page: Any = None
    metadata: Dict[str, Any]

class Source(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    llama_answer: str
    model_used: str
    grounded: bool
    retrieved_chunks: List[RetrievedChunk]
    sources: Optional[List[Source]] = None

class SimpleAnswerResponse(BaseModel):
    question: str
    llama_answer: str
    model_used: str
    grounded: bool

class EvaluationResult(BaseModel):
    summary: Dict[str, Any]
    details: List[Dict[str, Any]]
    ocr_results: List[Dict[str, Any]]