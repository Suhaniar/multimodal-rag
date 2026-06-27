import ollama
from fastapi import HTTPException
from app.core.config import OLLAMA_LLM, MAX_CONTEXT_CHARS
import logging

logger = logging.getLogger("multimodal_rag")

SYSTEM_PROMPT = """You are a precise helpful assistant.
Answer strictly from the provided document context.
If the answer is not present in the context, say exactly: I could not find this in the indexed documents.
Do not make up facts.
When the context contains a table with quarters and numbers, extract the exact number for the requested quarter.
Write a normal final answer, not just copied chunks."""

def generate_answer(context: str, question: str) -> str:
    context = context[:MAX_CONTEXT_CHARS]
    try:
        response = ollama.chat(
            model=OLLAMA_LLM,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ]
        )
    except Exception as exc:
        logger.error(f"Ollama LLM error: {exc}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Ollama error: {exc}")
    return response["message"]["content"].strip()

def rewrite_query(question: str) -> str:
    """Rewrite the query to be more retrieval-friendly."""
    prompt = f"Rephrase the following question to be more specific and keyword-rich for a search system. Only output the rewritten question:\n{question}"
    try:
        response = ollama.chat(
            model=OLLAMA_LLM,
            messages=[{"role": "user", "content": prompt}]
        )
        rewritten = response["message"]["content"].strip()
        return rewritten if rewritten else question
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return question