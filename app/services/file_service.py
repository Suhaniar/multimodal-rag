import uuid
import shutil
import tempfile
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException
from langchain_community.vectorstores import FAISS

from app.core.config import (
    SUPPORTED_EXTENSIONS, PREVIEW_CHARS, STORAGE_DIR, INDEXED_FILES_LIST_FILE
)
import app.core.state as state  # Always reference state.* directly — never import variables by value
from app.core.embeddings import embeddings
from app.services.ingestion_service import parse_upload
from app.services.retrieval_service import rebuild_page_index, rebuild_bm25_index

logger = logging.getLogger("multimodal_rag")


class FileService:
    async def ingest_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """Process uploaded files and add them to the index."""
        if not files:
            raise HTTPException(status_code=400, detail="Please upload at least one file.")

        for file in files:
            if file.size == 0:
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is empty.")

        logger.info(f"Ingestion started: {len(files)} file(s)")
        start_time = time.time()

        all_documents = []
        uploaded_files_info = []

        for file in files:
            if not file.filename:
                continue

            ext = Path(file.filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )

            with tempfile.TemporaryDirectory(prefix="multimodal_rag_") as temp_dir:
                file_path = Path(temp_dir) / file.filename
                with file_path.open("wb") as out:
                    shutil.copyfileobj(file.file, out)

                try:
                    doc_id = str(uuid.uuid4())
                    docs = parse_upload(file_path, file.filename, doc_id)
                    all_documents.extend(docs)

                    if file.filename in state.document_store:
                        state.document_store[file.filename].extend(docs)
                    else:
                        state.document_store[file.filename] = docs

                    preview_text = "\n\n".join(doc.page_content for doc in docs)[:PREVIEW_CHARS]
                    uploaded_files_info.append({"name": file.filename, "preview": preview_text})
                    logger.info(f"Processed file: {file.filename} -> {len(docs)} chunks")

                except Exception as exc:
                    logger.error(f"Error processing {file.filename}: {exc}", exc_info=True)
                    raise HTTPException(status_code=400, detail=f"Error processing {file.filename}: {str(exc)}")
                finally:
                    await file.close()

        if not all_documents:
            raise HTTPException(status_code=400, detail="No documents could be indexed.")

        # Build or update FAISS index
        if state.vectorstore is None:
            state.vectorstore = FAISS.from_documents(all_documents, embeddings)
        else:
            state.vectorstore.add_documents(all_documents)

        # Update indexed_files list (avoid duplicates)
        for info in uploaded_files_info:
            state.indexed_files[:] = [f for f in state.indexed_files if f["name"] != info["name"]]
            state.indexed_files.append(info)

        rebuild_page_index()
        rebuild_bm25_index()
        state.save_state()

        elapsed = time.time() - start_time
        logger.info(f"Ingestion completed in {elapsed:.2f}s, total documents indexed: {len(all_documents)}")

        combined_preview = "\n\n".join(info["preview"] for info in uploaded_files_info)[:PREVIEW_CHARS * 2]

        return {
            "message": "Files added successfully.",
            "files": [info["name"] for info in uploaded_files_info],
            "documents_indexed": len(all_documents),
            "modalities": sorted({doc.metadata.get("modality", "unknown") for doc in all_documents}),
            "total_vectors": state.vectorstore.index.ntotal,
            "indexed_files": [f["name"] for f in state.indexed_files],
            "extraction_preview": combined_preview,
            "ingestion_time_seconds": round(elapsed, 2),
        }

    def delete_files(self, filenames: List[str]) -> Dict[str, Any]:
        """Delete specified files from the index."""
        if state.vectorstore is None:
            raise HTTPException(status_code=400, detail="No files indexed yet.")

        logger.info(f"Deletion requested: {filenames}")

        deleted = []
        not_found = []
        for fname in filenames:
            if any(f["name"] == fname for f in state.indexed_files):
                state.indexed_files[:] = [f for f in state.indexed_files if f["name"] != fname]
                state.document_store.pop(fname, None)
                deleted.append(fname)
            else:
                not_found.append(fname)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"None of the specified files found. Not found: {not_found}"
            )

        remaining_docs = []
        for docs in state.document_store.values():
            remaining_docs.extend(docs)

        if not remaining_docs:
            state.vectorstore = None
            state.indexed_files.clear()
            state.document_store.clear()
            state.page_index.clear()
            state.bm25_index = None
            state.bm25_documents = []
            shutil.rmtree(STORAGE_DIR, ignore_errors=True)
            if INDEXED_FILES_LIST_FILE.exists():
                INDEXED_FILES_LIST_FILE.unlink()
            logger.info("All files deleted, index cleared.")
        else:
            state.vectorstore = FAISS.from_documents(remaining_docs, embeddings)
            rebuild_page_index()
            rebuild_bm25_index()

        state.save_state()

        logger.info(f"Deleted {len(deleted)} file(s). Remaining: {[f['name'] for f in state.indexed_files]}")
        return {
            "message": f"Deleted {len(deleted)} file(s). Index rebuilt.",
            "deleted": deleted,
            "not_found": not_found,
            "remaining_files": [f["name"] for f in state.indexed_files],
        }


file_service = FileService()