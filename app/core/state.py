import os
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from .config import FAISS_INDEX_DIR, STORAGE_DIR, INDEXED_FILES_LIST_FILE
from .embeddings import embeddings
from app.helpers.text_utils import tokenize_bm25
import logging

logger = logging.getLogger("multimodal_rag")

# Global state — always access via state.* (never import these variables directly)
vectorstore: FAISS | None = None
indexed_files: List[Dict[str, str]] = []
document_store: Dict[str, List[Document]] = {}
page_index: Dict[Tuple[str, int], List[Document]] = {}
bm25_index: BM25Okapi | None = None
bm25_documents: List[Document] = []
reranker: CrossEncoder | None = None


def save_state():
    """Persist all state to disk."""
    os.makedirs(STORAGE_DIR, exist_ok=True)

    if vectorstore is not None:
        try:
            vectorstore.save_local(str(FAISS_INDEX_DIR))
            logger.info(f"Saved FAISS index ({vectorstore.index.ntotal} vectors)")
        except Exception as e:
            logger.error(f"Failed to save FAISS: {e}")

    try:
        with open(STORAGE_DIR / "document_store.pkl", "wb") as f:
            pickle.dump(document_store, f)
        logger.info(f"Saved document_store ({len(document_store)} files)")
    except Exception as e:
        logger.error(f"Failed saving document_store: {e}")

    try:
        with open(STORAGE_DIR / "page_index.pkl", "wb") as f:
            pickle.dump(page_index, f)
        logger.info(f"Saved page_index ({len(page_index)} pages)")
    except Exception as e:
        logger.error(f"Failed saving page_index: {e}")

    try:
        with open(STORAGE_DIR / "bm25_documents.pkl", "wb") as f:
            pickle.dump(bm25_documents, f)
        logger.info(f"Saved BM25 documents ({len(bm25_documents)} chunks)")
    except Exception as e:
        logger.error(f"Failed saving BM25 documents: {e}")

    try:
        with open(INDEXED_FILES_LIST_FILE, "w") as f:
            json.dump(indexed_files, f, indent=2)
        logger.info(f"Saved indexed_files ({len(indexed_files)} files)")
    except Exception as e:
        logger.error(f"Failed saving indexed files: {e}")

    logger.info("Application state saved successfully.")


def load_state():
    """Load persisted state from disk on startup."""
    global vectorstore, document_store, page_index, bm25_index, bm25_documents, indexed_files

    # --- FAISS ---
    if FAISS_INDEX_DIR.exists() and any(FAISS_INDEX_DIR.iterdir()):
        try:
            vectorstore = FAISS.load_local(
                str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded FAISS index from '{FAISS_INDEX_DIR}'")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}. Starting fresh.")
            vectorstore = None
    else:
        vectorstore = None

    # --- Document store ---
    doc_store_path = STORAGE_DIR / "document_store.pkl"
    if doc_store_path.exists():
        try:
            with open(doc_store_path, "rb") as f:
                document_store = pickle.load(f)
            logger.info(f"Loaded document_store with {len(document_store)} files.")
        except Exception as e:
            logger.warning(f"Failed to load document_store: {e}. Starting empty.")
            document_store = {}
    else:
        document_store = {}

    # --- Page index ---
    page_idx_path = STORAGE_DIR / "page_index.pkl"
    if page_idx_path.exists():
        try:
            with open(page_idx_path, "rb") as f:
                page_index = pickle.load(f)
            logger.info(f"Loaded page_index with {len(page_index)} entries.")
        except Exception as e:
            logger.warning(f"Failed to load page_index: {e}. Starting empty.")
            page_index = {}
    else:
        page_index = {}

    # --- BM25 ---
    bm25_docs_path = STORAGE_DIR / "bm25_documents.pkl"
    if bm25_docs_path.exists():
        try:
            with open(bm25_docs_path, "rb") as f:
                bm25_documents = pickle.load(f)
            if bm25_documents:
                tokenized_corpus = [tokenize_bm25(doc.page_content) for doc in bm25_documents]
                bm25_index = BM25Okapi(tokenized_corpus)
                logger.info(f"Loaded and rebuilt BM25 index with {len(bm25_documents)} documents.")
            else:
                bm25_index = None
                bm25_documents = []
        except Exception as e:
            logger.warning(f"Failed to load bm25_documents: {e}. Starting empty.")
            bm25_index = None
            bm25_documents = []
    else:
        bm25_index = None
        bm25_documents = []

    # --- Indexed files list ---
    if INDEXED_FILES_LIST_FILE.exists():
        try:
            with open(INDEXED_FILES_LIST_FILE, "r") as f:
                loaded = json.load(f)
            indexed_files = []
            for item in loaded:
                if isinstance(item, str):
                    indexed_files.append({"name": item, "preview": ""})
                elif isinstance(item, dict):
                    indexed_files.append({
                        "name": item.get("name") or item.get("filename", "unknown"),
                        "preview": item.get("preview", ""),
                    })
            logger.info(f"Loaded indexed files list: {indexed_files}")
        except Exception as e:
            logger.warning(f"Could not load indexed_files.json: {e}")
            indexed_files = []
    else:
        indexed_files = []

    # --- Fallback: rebuild BM25 from document_store if pkl was missing ---
    if bm25_index is None and document_store:
        all_docs = []
        for docs in document_store.values():
            all_docs.extend(docs)
        if all_docs:
            bm25_documents = all_docs
            tokenized = [tokenize_bm25(doc.page_content) for doc in all_docs]
            bm25_index = BM25Okapi(tokenized)
            logger.info(f"Rebuilt BM25 from document_store ({len(all_docs)} documents)")