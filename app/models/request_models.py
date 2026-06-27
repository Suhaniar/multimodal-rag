from pydantic import BaseModel, Field
from typing import List
from app.core.config import TOP_K, HYBRID_ALPHA

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=TOP_K, ge=1, le=50, description="Number of top results to return")
    include_sources: bool = True
    alpha: float = Field(default=HYBRID_ALPHA, ge=0.0, le=1.0, description="Weight for FAISS vs BM25 (0=BM25 only, 1=FAISS only)")

class DeleteRequest(BaseModel):
    filenames: List[str]