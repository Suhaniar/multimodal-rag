<div align="center">

# 🧠 Multimodal RAG Backend

**A production-ready Retrieval-Augmented Generation system with hybrid search, cross-encoder re-ranking, and multimodal enrichment.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Dense%20Search-4B8BBE?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-LLM%20%2B%20OCR-black?style=flat-square)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[Features](#-features) · [Tech Stack](#-tech-stack) · [Installation](#-installation) · [API Reference](#-api-endpoints) · [Configuration](#-configuration-reference) · [Evaluation](#-evaluation)

</div>

---

## ✨ Features

| Category | Capability |
|----------|-----------|
| 📄 **Multi-format Ingestion** | PDF (text + OCR + images + tables), images (OCR + captioning), DOCX, PPTX, CSV, TXT/MD |
| 🔍 **Hybrid Retrieval** | FAISS (dense) + BM25 (sparse) with configurable alpha — static or dynamic |
| 🎯 **Re-ranking** | Cross-encoder re-ranking via sentence-transformers for improved relevance |
| 🖼️ **Multimodal Enrichment** | Image captions, object lists, scene summaries, chart descriptions |
| 🏗️ **Parent-Child Chunking** | Preserves document hierarchy; retrieves full parent contexts |
| 📑 **Section Hierarchy** | Extracts headings and builds section paths for richer context |
| ⚡ **Dynamic Alpha** | Auto-adjusts hybrid FAISS/BM25 weight based on query length |
| ✍️ **Query Rewriting** | Rephrases user questions before retrieval for better results |
| 💾 **Full Persistence** | FAISS index, document store, page index, and BM25 saved to disk |
| 📊 **Evaluation Suite** | Built-in recall, accuracy, grounding, and OCR accuracy tests |
| 📝 **Production Logging** | Rotating file logs + console output |

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| REST API | [FastAPI](https://fastapi.tiangolo.com/) |
| Document handling & embeddings | [LangChain](https://langchain.com/) |
| Dense vector search | [FAISS](https://github.com/facebookresearch/faiss) |
| Sparse keyword search | [rank_bm25](https://github.com/dorianbrown/rank_bm25) |
| Embeddings & cross-encoder | [sentence-transformers](https://www.sbert.net/) |
| LLM + OCR | [Ollama](https://ollama.com/) (`llama3.2` / `deepseek-ocr`) |
| PDF extraction | [PyMuPDF](https://pymupdf.readthedocs.io/), [pdfplumber](https://github.com/jsvine/pdfplumber) |
| Image processing | [Pillow](https://python-pillow.org/) |
| DOCX / PPTX extraction | [python-docx](https://python-docx.readthedocs.io/), [python-pptx](https://python-pptx.readthedocs.io/) |
| Evaluation metric | [Levenshtein](https://github.com/ztane/python-Levenshtein) |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Suhaniar/multimodal-rag
cd multimodal-rag

```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

Download from [ollama.com](https://ollama.com), then pull the required models:

```bash
ollama pull llama3.2
ollama pull deepseek-ocr:latest
```

> Keep Ollama running in the background with `ollama serve`.

### 5. Configure environment
```bash
backend.env has all configurations.They can be changed as per necessity

```

## 🚀 Running the Server

```bash
uvicorn app.main:app --reload
```

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | API base URL |
| `http://localhost:8000/docs` | Interactive Swagger UI |

---

## 📡 API Endpoints

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload one or more files (PDF, image, DOCX, PPTX, CSV, TXT) for indexing |
| `DELETE` | `/delete` | Delete previously indexed files by name |
| `POST` | `/query` | Ask a question; returns answer, retrieved chunks, and optional sources |
| `POST` | `/answer` | Ask a question; returns only the LLM answer and grounding flag |
| `POST` | `/evaluate` | Run the evaluation suite against `evaluation_set.json` |
| `GET` | `/health` | Check system status (Ollama, FAISS, BM25, reranker) |

### Debug

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/debug/page_index` | Inspect the page index |
| `GET` | `/debug/bm25_status` | Check BM25 readiness |
| `GET` | `/debug/reranker_status` | Check reranker readiness |
| `POST` | `/debug/retrieve` | Inspect raw hybrid + cross-encoder scores for a query |

---

## 🧪 Evaluation

Create an `evaluation_set.json` file in the project root:

```json
[
  {
    "question": "What is the revenue in Q2 2024?",
    "expected_answer": "$12.5 million",
    "relevant_source": "financial_report.pdf",
    "ocr_test": {
      "file_path": "samples/sample.pdf",
      "page": 3,
      "ground_truth_text": "Revenue Q2: $12.5M"
    }
  }
]
```

Then trigger the suite:

```bash
POST /evaluate
```

Results include **retrieval recall**, **answer accuracy** (exact/partial), **grounding rate**, and **OCR similarity**.

---

## 💾 Persistence

All indexes and metadata are stored automatically and reloaded on startup:

| Path | Contents |
|------|----------|
| `faiss_index/` | FAISS vector index |
| `storage/document_store.pkl` | Raw document chunks |
| `storage/page_index.pkl` | Page-level document map |
| `storage/bm25_documents.pkl` | BM25 document list |
| `indexed_files.json` | List of ingested filenames |

---

## 📁 Folder Structure

```
pdf_chatbot/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # API endpoints (delegates to services)
│   ├── core/
│   │   ├── config.py               # All env-based configuration
│   │   ├── embeddings.py           # Embedding model setup
│   │   ├── startup.py              # Lifespan / startup logic
│   │   └── state.py                # Global mutable state
│   ├── helpers/
│   │   ├── image_utils.py
│   │   └── text_utils.py
│   ├── models/
│   │   ├── request_models.py
│   │   └── response_models.py
│   └── services/
│       ├── evaluation_service.py
│       ├── file_service.py         # Ingest + delete logic
│       ├── ingestion_service.py
│       ├── llm_service.py
│       ├── ocr_service.py
│       └── retrieval_service.py
├── backend.env                     # Environment variables
├── main.py                         # App entry point
├── evaluation_set.json             # (optional) evaluation dataset
├── indexed_files.json              # (runtime) ingested file list
├── faiss_index/                    # (runtime) vector index
├── storage/                        # (runtime) persisted state
└── logs/                           # (runtime) log files
```

---

## ⚙️ Configuration Reference

All settings live in `backend.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `llama3.2` | Ollama model for answering |
| `OCR_MODEL` | `deepseek-ocr:latest` | Ollama model for OCR |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `CHUNK_SIZE` | `500` | Token size per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `HYBRID_ALPHA` | `0.7` | FAISS vs BM25 weight (`1.0` = FAISS only) |
| `DYNAMIC_ALPHA` | `true` | Auto-adjust alpha by query length |
| `ENABLE_QUERY_REWRITE` | `true` | Rewrite queries before retrieval |
| `ENABLE_ENRICHMENT` | `true` | Enable image/chart enrichment |

---

## 📋 Logging

| Target | Level | Details |
|--------|-------|---------|
| Console | `INFO` | Live output |
| `logs/rag_backend.log` | `INFO` | Rotating, max 10 MB, 5 backups |

---

## 🔧 Troubleshooting

<details>
<summary><b>Ollama not reachable</b></summary>

Run `ollama serve` and confirm models are pulled:

```bash
ollama list
```
</details>

<details>
<summary><b>FAISS index not loading</b></summary>

Delete the index directory and restart — the system rebuilds from `storage/`:

```bash
rm -rf faiss_index/
uvicorn app.main:app --reload
```
</details>

<details>
<summary><b>Import errors</b></summary>

Ensure the virtual environment is active and all dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary><b>Empty responses</b></summary>

Check `/health` to confirm all components are ready:

```json
{
  "index_ready": true,
  "bm25_ready": true,
  "reranker_ready": true
}
```
</details>

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

 [Report a Bug](../../issues) · [Request a Feature](../../issues)

</div>
