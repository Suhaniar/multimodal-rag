import time
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from fastapi import HTTPException

from app.core.config import (
    CANDIDATE_MULTIPLIER,
    CE_WEIGHT,
    HYBRID_ALPHA,
    TOP_K,
    DYNAMIC_ALPHA,
    ENABLE_QUERY_REWRITE,
)
import app.core.state as state
from app.core.embeddings import embeddings
from app.helpers.text_utils import document_key, tokenize_bm25
from app.services.llm_service import generate_answer, rewrite_query
from app.models.request_models import QueryRequest
from app.models.response_models import RetrievedChunk, Source

logger = logging.getLogger("multimodal_rag")

# ---------- BM25 & Page Index rebuild ----------
def rebuild_bm25_index():
    all_docs = []
    for docs in state.document_store.values():
        all_docs.extend(docs)
    if not all_docs:
        state.bm25_index = None
        state.bm25_documents = []
        return
    tokenized = [tokenize_bm25(doc.page_content) for doc in all_docs]
    state.bm25_documents = all_docs
    state.bm25_index = BM25Okapi(tokenized)
    logger.info(f"BM25 rebuilt with {len(all_docs)} documents")

def rebuild_page_index():
    state.page_index.clear()
    for filename, docs in state.document_store.items():
        for doc in docs:
            page = doc.metadata.get("page") or doc.metadata.get("slide")
            if page is None:
                continue
            key = (filename, page)
            state.page_index.setdefault(key, []).append(doc)
    logger.info(f"Page index rebuilt ({len(state.page_index)} pages)")

# ---------- Retrieval core ----------
def retrieve_candidates(question: str, candidate_k: int, alpha: float = HYBRID_ALPHA):
    if state.vectorstore is None and state.bm25_index is None:
        raise HTTPException(status_code=400, detail="No files indexed yet.")

    faiss_results = []
    if state.vectorstore is not None:
        try:
            raw_faiss = state.vectorstore.similarity_search_with_score(question, k=candidate_k * 2)
            faiss_results = [(doc, 1 / (1 + distance)) for doc, distance in raw_faiss]
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Embedding search failed")

    bm25_results = []
    if state.bm25_index is not None:
        tokenized_query = tokenize_bm25(question)
        bm25_scores = state.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_k * 2]
        bm25_results = [(state.bm25_documents[i], bm25_scores[i]) for i in top_indices]

    all_docs = {}
    for doc, sim in faiss_results:
        all_docs[document_key(doc)] = (doc, sim, None)
    for doc, score in bm25_results:
        key = document_key(doc)
        if key in all_docs:
            old_doc, faiss_sim, _ = all_docs[key]
            all_docs[key] = (old_doc, faiss_sim, score)
        else:
            all_docs[key] = (doc, None, score)

    faiss_scores = [x[1] for x in all_docs.values() if x[1] is not None]
    bm25_scores = [x[2] for x in all_docs.values() if x[2] is not None]
    max_faiss = max(faiss_scores) if faiss_scores else 1
    max_bm25 = max(bm25_scores) if bm25_scores else 1

    combined = []
    for doc, faiss_score, bm25_score in all_docs.values():
        norm_faiss = faiss_score / max_faiss if faiss_score is not None else 0
        norm_bm25 = bm25_score / max_bm25 if bm25_score is not None else 0
        score = alpha * norm_faiss + (1 - alpha) * norm_bm25
        combined.append((doc, score))
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:candidate_k]

def rerank_documents(question, candidates, top_k):
    if not candidates or state.reranker is None:
        return candidates[:top_k]
    pairs = [(question, doc.page_content) for doc, _ in candidates]
    ce_scores = state.reranker.predict(pairs)
    mn = min(ce_scores)
    mx = max(ce_scores)
    if mx - mn > 1e-8:
        ce_scores = [(x - mn) / (mx - mn) for x in ce_scores]
    else:
        ce_scores = [0.5] * len(ce_scores)
    results = []
    for (doc, hybrid), ce in zip(candidates, ce_scores):
        final_score = (1 - CE_WEIGHT) * hybrid + CE_WEIGHT * ce
        results.append((doc, final_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def deduplicate_by_numeric_content(docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    page_best = {}
    for doc, score in docs:
        page = doc.metadata.get("page") or doc.metadata.get("slide")
        if page is None:
            continue
        key = (doc.metadata.get("source"), page)
        num_digits = sum(c.isdigit() for c in doc.page_content)
        if key not in page_best or num_digits > page_best[key][1]:
            page_best[key] = (doc, score, num_digits)
    result = []
    seen_pages = set()
    for doc, score in docs:
        page = doc.metadata.get("page") or doc.metadata.get("slide")
        if page is None:
            result.append((doc, score))
        else:
            key = (doc.metadata.get("source"), page)
            if key not in seen_pages:
                best_doc, best_score, _ = page_best[key]
                result.append((best_doc, best_score))
                seen_pages.add(key)
    return result

# ---------- Dynamic alpha (Phase 2) ----------
def compute_dynamic_alpha(question: str) -> float:
    """
    Adjust alpha based on query length:
    - Short queries (<5 words) → lean BM25 (alpha lower)
    - Long queries (>7 words) → lean FAISS (alpha higher)
    Base value from HYBRID_ALPHA.
    """
    base = HYBRID_ALPHA
    words = question.split()
    num_words = len(words)
    if num_words <= 3:
        return max(0.2, base - 0.3)
    elif num_words <= 6:
        return base
    else:
        return min(0.9, base + 0.2)

# ---------- FULL retrieve_and_answer WITH PARENT-CHILD + DYNAMIC ALPHA ----------
def retrieve_and_answer(
    request: QueryRequest,
    include_cross_modal: bool = True,
    max_neighbors_per_page: int = 3
):
    from app.core.state import page_index

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ---- Phase 2: Query rewriting ----
    if ENABLE_QUERY_REWRITE:
        rewritten = rewrite_query(question)
        if rewritten and rewritten != question:
            logger.info(f"Query rewritten: '{question}' → '{rewritten}'")
            question = rewritten

    # ---- Phase 2: Dynamic alpha ----
    alpha = request.alpha
    if DYNAMIC_ALPHA and alpha == HYBRID_ALPHA:
        alpha = compute_dynamic_alpha(question)
        logger.info(f"Dynamic alpha set to {alpha:.2f}")

    logger.info(f"Query received: '{question}' (top_k={request.top_k}, alpha={alpha:.2f})")
    start_time = time.time()

    candidate_k = request.top_k * CANDIDATE_MULTIPLIER
    hybrid_candidates = retrieve_candidates(question, candidate_k, alpha)
    if not hybrid_candidates:
        logger.info(f"Query completed (no results) in {time.time() - start_time:.3f}s")
        return "I could not find this in the indexed documents.", False, [], None

    reranked = rerank_documents(question, hybrid_candidates, request.top_k)

    # ---------- Step 1: collect top chunks + cross modal neighbors ----------
    final_docs: List[Tuple[Document, float]] = []
    seen_keys = set()
    for doc, score in reranked:
        key = document_key(doc)
        if key not in seen_keys:
            seen_keys.add(key)
            final_docs.append((doc, score))

        if include_cross_modal:
            source = doc.metadata.get("source")
            page = doc.metadata.get("page") or doc.metadata.get("slide")
            if source and page is not None:
                page_key = (source, page)
                neighbors = page_index.get(page_key, [])
                for neighbor in neighbors:
                    nkey = document_key(neighbor)
                    if nkey not in seen_keys:
                        seen_keys.add(nkey)
                        final_docs.append((neighbor, score * 0.9))
                        if len(final_docs) >= (request.top_k + max_neighbors_per_page * len(reranked)):
                            break
        if len(final_docs) >= (request.top_k + max_neighbors_per_page * len(reranked)):
            break

    # ---------- Step 2: add parent documents (Phase 2) ----------
    parent_docs = []
    seen_parent_ids = set()
    for doc, score in final_docs:
        parent_id = doc.metadata.get("parent_document_id")
        if parent_id and parent_id not in seen_parent_ids:
            for docs in state.document_store.values():
                for d in docs:
                    if (d.metadata.get("document_id") == parent_id and
                        d.metadata.get("kind") == "document"):
                        parent_docs.append((d, score * 0.8))
                        seen_parent_ids.add(parent_id)
                        break
                if parent_id in seen_parent_ids:
                    break

    combined = final_docs + parent_docs
    seen = set()
    unique = []
    for doc, score in combined:
        key = document_key(doc)
        if key not in seen:
            seen.add(key)
            unique.append((doc, score))

    final_docs = unique[:request.top_k * 2]

    # ---------- Step 3: final context & answer ----------
    final_docs = deduplicate_by_numeric_content(final_docs)
    context = "\n\n".join(doc.page_content for doc, _ in final_docs)
    grounded = len(final_docs) > 0
    llama_answer = generate_answer(context, question) if grounded else "I could not find this in the indexed documents."

    elapsed = time.time() - start_time
    logger.info(f"Query completed in {elapsed:.3f}s, grounded={grounded}, retrieved={len(final_docs)} chunks")

    retrieved_chunks = [
        RetrievedChunk(
            chunk=doc.page_content,
            score=float(score),
            source=doc.metadata.get("source", "unknown"),
            modality=doc.metadata.get("modality", "unknown"),
            page=doc.metadata.get("page") or doc.metadata.get("slide"),
            metadata=doc.metadata
        ) for doc, score in final_docs
    ]

    sources = None
    if request.include_sources:
        sources = [
            Source(content=doc.page_content, score=float(score), metadata=doc.metadata)
            for doc, score in final_docs
        ]

    return llama_answer, grounded, retrieved_chunks, sources