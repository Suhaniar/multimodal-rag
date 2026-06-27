import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("backend.env")   # Load from backend.env

# Application
APP_TITLE = os.getenv("APP_TITLE", "Multimodal RAG Backend")

# Models
OLLAMA_LLM = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_OCR_MODEL = os.getenv("OCR_MODEL", "deepseek-ocr:latest")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))
CANDIDATE_MULTIPLIER = int(os.getenv("CANDIDATE_MULTIPLIER", "4"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

# OCR
OCR_ZOOM = int(os.getenv("OCR_ZOOM", "2"))
OCR_MIN_NATIVE_TEXT_CHARS = int(os.getenv("OCR_MIN_NATIVE_TEXT_CHARS", "80"))
OCR_MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", "4"))
OCR_MAX_IMAGE_SIZE = int(os.getenv("OCR_MAX_IMAGE_SIZE", "800"))
PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "200"))

# Persistence
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "faiss_index"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
INDEXED_FILES_LIST_FILE = Path("indexed_files.json")   # kept for backward compatibility

# Cross-encoder blending
CE_WEIGHT = float(os.getenv("CE_WEIGHT", "0.5"))

# Enrichment
ENABLE_ENRICHMENT = os.getenv("ENABLE_ENRICHMENT", "true").lower() in ("true", "1", "yes")

# Evaluation file
# Evaluation dataset
EVALUATION_SET_FILE = Path(
    os.getenv(
        "EVALUATION_SET_FILE",
        "evaluation_set.json"
    )
)
# Supported extensions – not configurable
SUPPORTED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    ".docx", ".pptx", ".txt", ".md", ".csv",
}

DYNAMIC_ALPHA = os.getenv("DYNAMIC_ALPHA", "true").lower() in ("true", "1", "yes")
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() in ("true", "1", "yes")