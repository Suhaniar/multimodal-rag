import uuid
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import ollama
import time
import logging

from app.core.config import (
    APP_TITLE, OLLAMA_LLM, OLLAMA_OCR_MODEL, RERANKER_MODEL,
    HYBRID_ALPHA, ENABLE_ENRICHMENT, SUPPORTED_EXTENSIONS,
    PREVIEW_CHARS, STORAGE_DIR, INDEXED_FILES_LIST_FILE,
    DYNAMIC_ALPHA, ENABLE_QUERY_REWRITE
)
import app.core.state as state
from app.models.request_models import QueryRequest, DeleteRequest
from app.models.response_models import QueryResponse, SimpleAnswerResponse, EvaluationResult
from app.services.retrieval_service import (
    retrieve_and_answer, rebuild_page_index, rebuild_bm25_index,
    compute_dynamic_alpha
)
from app.services.evaluation_service import evaluate_system
from app.services.file_service import file_service  # <-- use the service

router = APIRouter()
logger = logging.getLogger("multimodal_rag")

# ---------- Debug endpoints (unchanged) ----------
@router.get("/debug/page_index")
def debug_page_index():
    result = {}
    for (src, page), docs in state.page_index.items():
        result[f"{src} page {page}"] = [
            {"modality": d.metadata.get("modality"), "preview": d.page_content[:80]}
            for d in docs
        ]
    return result

@router.get("/debug/bm25_status")
def debug_bm25_status():
    return {"bm25_ready": state.bm25_index is not None, "num_bm25_docs": len(state.bm25_documents) if state.bm25_documents else 0}

@router.get("/debug/reranker_status")
def debug_reranker_status():
    return {"reranker_loaded": state.reranker is not None}

@router.post("/debug/retrieve")
def debug_retrieve(request: QueryRequest):
    from app.services.retrieval_service import retrieve_candidates
    question = request.question.strip()
    candidate_k = request.top_k * 4
    hybrid_candidates = retrieve_candidates(question, candidate_k, request.alpha)
    if hybrid_candidates and state.reranker is not None:
        pairs = [(question, doc.page_content) for doc, _ in hybrid_candidates[:request.top_k]]
        ce_raw = state.reranker.predict(pairs)
        min_ce, max_ce = min(ce_raw), max(ce_raw)
        ce_scaled = [(s - min_ce) / (max_ce - min_ce) if max_ce - min_ce > 1e-8 else 0.5 for s in ce_raw]
    else:
        ce_raw, ce_scaled = [], []
    return {
        "question": question,
        "hybrid_scores": [(doc.page_content[:80], score) for doc, score in hybrid_candidates[:request.top_k]],
        "cross_encoder_raw": ce_raw[:request.top_k],
        "cross_encoder_scaled": ce_scaled[:request.top_k],
    }

# ---------- Main endpoints ----------
@router.get("/")
def root():
    return {
        "message": f"{APP_TITLE} with parallel OCR, cross-modal linking, hybrid retrieval (FAISS+BM25), cross-encoder re-ranking, production logging, multimodal enrichment, full persistence, OCR fix, parameter validation, metadata duplication fix, parent–child hierarchy, section hierarchy, table summarization, chart understanding, dynamic hybrid weighting, and query rewriting.",
        "llm": OLLAMA_LLM,
        "ocr_model": OLLAMA_OCR_MODEL,
        "reranker_model": RERANKER_MODEL,
        "hybrid_alpha": HYBRID_ALPHA,
        "enrichment_enabled": ENABLE_ENRICHMENT,
        "docs": "/docs",
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }

@router.get("/health")
def health():
    ollama_status = "ok"
    try:
        ollama.list()
    except Exception as exc:
        ollama_status = f"unreachable: {exc}"
        logger.error(f"Ollama health check failed: {exc}")
    return {
        "status": "ok",
        "llm": OLLAMA_LLM,
        "ocr_model": OLLAMA_OCR_MODEL,
        "ollama": ollama_status,
        "indexed_files": state.indexed_files,
        "index_ready": state.vectorstore is not None,
        "bm25_ready": state.bm25_index is not None,
        "reranker_ready": state.reranker is not None,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }

@router.post("/ingest", openapi_extra={
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string", "format": "binary"}}
                    },
                    "required": ["files"]
                }
            }
        }
    }
})
async def ingest_files(files: List[UploadFile] = File(...)):
    """Upload one or more files (PDF, image, DOCX, PPTX, CSV, TXT) for indexing."""
    return await file_service.ingest_files(files)

@router.delete("/delete")
async def delete_files(request: DeleteRequest):
    return file_service.delete_files(request.filenames)

@router.post("/answer", response_model=SimpleAnswerResponse)
def answer_only(request: QueryRequest):
    llama_answer, grounded, _, _ = retrieve_and_answer(request)
    return SimpleAnswerResponse(question=request.question, llama_answer=llama_answer, model_used=OLLAMA_LLM, grounded=grounded)

@router.post("/query", response_model=QueryResponse)
def query_index(request: QueryRequest):
    # Apply dynamic alpha if enabled and user didn't override
    if DYNAMIC_ALPHA and request.alpha == HYBRID_ALPHA:
        request.alpha = compute_dynamic_alpha(request.question)

    llama_answer, grounded, retrieved_chunks, sources = retrieve_and_answer(request)

    return QueryResponse(
        llama_answer=llama_answer,
        model_used=OLLAMA_LLM,
        grounded=grounded,
        retrieved_chunks=retrieved_chunks,
        sources=sources
    )

@router.post("/evaluate", response_model=EvaluationResult)
def evaluate():
    return evaluate_system()