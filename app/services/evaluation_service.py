import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from fastapi import HTTPException
from app.core.config import TOP_K, HYBRID_ALPHA, OCR_ZOOM
from app.models.request_models import QueryRequest
from app.models.response_models import EvaluationResult
from app.services.retrieval_service import retrieve_and_answer
from app.services.ocr_service import run_ocr
import Levenshtein
from PIL import Image
import pymupdf as fitz

logger = logging.getLogger("multimodal_rag")

def load_evaluation_set() -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset.

    Priority:
    1. File specified in backend.env via EVALUATION_SET_FILE
    2. Common fallback filenames
    """

    from app.core.config import EVALUATION_SET_FILE

    candidate_files = [
        EVALUATION_SET_FILE,
        Path("evaluation_set.json"),
        Path("evaluation_dataset.json"),
        Path("eval.json"),
        Path("eval_set.json"),
        Path("tests.json"),
        Path("test_cases.json"),
    ]

    # Remove duplicate paths while preserving order
    seen = set()
    unique_files = []

    for path in candidate_files:
        path = Path(path)
        if path not in seen:
            unique_files.append(path)
            seen.add(path)

    for path in unique_files:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    raise ValueError(
                        f"{path.name} must contain a JSON array."
                    )

                logger.info(
                    f"Loaded evaluation dataset from {path}"
                )

                return data

            except Exception as e:
                logger.warning(
                    f"Failed loading {path}: {e}"
                )

    tried = ", ".join(str(p) for p in unique_files)

    raise HTTPException(
        status_code=400,
        detail=(
            "No evaluation dataset found.\n"
            f"Tried: {tried}"
        ),
    )
def compute_retrieval_recall(question: str, relevant_source: str, top_k: int = TOP_K) -> bool:
    class DummyRequest:
        def __init__(self, q, k, alpha):
            self.question = q
            self.top_k = k
            self.alpha = alpha
            self.include_sources = False
    request = DummyRequest(question, top_k, HYBRID_ALPHA)
    try:
        _, _, retrieved_chunks, _ = retrieve_and_answer(request)
        retrieved_sources = {chunk.source for chunk in retrieved_chunks}
        return relevant_source in retrieved_sources
    except Exception as e:
        logger.error(f"Retrieval recall evaluation failed for '{question}': {e}")
        return False

def compute_answer_accuracy(question: str, expected_answer: str) -> Tuple[bool, bool]:
    class DummyRequest:
        def __init__(self, q, k, alpha):
            self.question = q
            self.top_k = k
            self.alpha = alpha
            self.include_sources = False
    request = DummyRequest(question, TOP_K, HYBRID_ALPHA)
    try:
        llama_answer, _, _, _ = retrieve_and_answer(request)
        norm_expected = expected_answer.strip().lower()
        norm_llama = llama_answer.strip().lower()
        exact = (norm_llama == norm_expected)
        partial = (norm_expected in norm_llama) if norm_expected else False
        return exact, partial
    except Exception as e:
        logger.error(f"Answer accuracy evaluation failed for '{question}': {e}")
        return False, False

def compute_grounding_rate(question: str) -> bool:
    class DummyRequest:
        def __init__(self, q, k, alpha):
            self.question = q
            self.top_k = k
            self.alpha = alpha
            self.include_sources = False
    request = DummyRequest(question, TOP_K, HYBRID_ALPHA)
    try:
        _, grounded, _, _ = retrieve_and_answer(request)
        return grounded
    except Exception as e:
        logger.error(f"Grounding evaluation failed for '{question}': {e}")
        return False

def levenshtein_distance(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return 1.0 - (Levenshtein.distance(s1, s2) / max(len(s1), len(s2)))

def get_ocr_text_for_file(file_path: str, page_num: int = 1) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"OCR test file not found: {file_path}")
    if path.suffix.lower() == ".pdf":
        pdf = fitz.open(path)
        if page_num < 1 or page_num > len(pdf):
            pdf.close()
            raise ValueError(f"Page {page_num} out of range for PDF {file_path}")
        page = pdf[page_num - 1]
        matrix = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        pdf.close()
        return run_ocr(image)
    elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return run_ocr(img)
    else:
        raise ValueError(f"OCR test only supports PDF or image files, got {path.suffix}")

def compute_ocr_accuracy(ocr_test_config: Dict[str, Any]) -> Dict[str, Any]:
    file_path = ocr_test_config.get("file_path")
    page = ocr_test_config.get("page", 1)
    ground_truth = ocr_test_config.get("ground_truth_text", "")
    if not file_path or not ground_truth:
        return {"error": "Missing file_path or ground_truth_text in ocr_test"}
    try:
        ocr_output = get_ocr_text_for_file(file_path, page)
        similarity = levenshtein_distance(ocr_output, ground_truth)
        return {
            "file": file_path,
            "page": page,
            "ground_truth": ground_truth[:200],
            "ocr_output": ocr_output[:200],
            "similarity": similarity,
            "success": True
        }
    except Exception as e:
        logger.error(f"OCR test failed: {e}")
        return {
            "file": file_path,
            "page": page,
            "error": str(e),
            "success": False
        }

def evaluate_system() -> EvaluationResult:
    test_cases = load_evaluation_set()
    if not test_cases:
        raise HTTPException(status_code=400, detail="Evaluation set is empty")

    logger.info(f"Starting evaluation with {len(test_cases)} test cases")
    start_time = time.time()

    retrieval_successes = 0
    answer_exact_successes = 0
    answer_partial_successes = 0
    grounding_successes = 0
    retrieval_total = 0
    answer_total = 0
    grounding_total = 0

    details = []
    ocr_results = []

    for idx, case in enumerate(test_cases):
        question = case.get("question", "")
        expected_answer = case.get("expected_answer", "")
        relevant_source = case.get("relevant_source")
        ocr_test = case.get("ocr_test")

        if not question:
            logger.warning(f"Test case {idx} missing 'question', skipping")
            continue

        recall_ok = False
        if relevant_source:
            retrieval_total += 1
            recall_ok = compute_retrieval_recall(question, relevant_source, TOP_K)
            if recall_ok:
                retrieval_successes += 1

        answer_exact = False
        answer_partial = False
        if expected_answer:
            answer_total += 1
            answer_exact, answer_partial = compute_answer_accuracy(question, expected_answer)
            if answer_exact:
                answer_exact_successes += 1
            if answer_partial:
                answer_partial_successes += 1

        grounding_total += 1
        grounded = compute_grounding_rate(question)
        if grounded:
            grounding_successes += 1

        detail = {
            "question": question,
            "expected_answer": expected_answer,
            "relevant_source": relevant_source,
            "retrieval_success": recall_ok if relevant_source else None,
            "answer_exact_match": answer_exact if expected_answer else None,
            "answer_partial_match": answer_partial if expected_answer else None,
            "grounded": grounded,
        }
        details.append(detail)

        if ocr_test:
            ocr_result = compute_ocr_accuracy(ocr_test)
            ocr_results.append(ocr_result)

    summary = {
        "total_queries": len(test_cases),
        "retrieval_recall": retrieval_successes / retrieval_total if retrieval_total > 0 else None,
        "answer_accuracy_exact": answer_exact_successes / answer_total if answer_total > 0 else None,
        "answer_accuracy_partial": answer_partial_successes / answer_total if answer_total > 0 else None,
        "grounding_rate": grounding_successes / grounding_total if grounding_total > 0 else None,
        "ocr_accuracy": None,
    }

    if ocr_results:
        valid_ocr = [r["similarity"] for r in ocr_results if r.get("success")]
        if valid_ocr:
            summary["ocr_accuracy"] = sum(valid_ocr) / len(valid_ocr)

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f}s: {summary}")

    return EvaluationResult(summary=summary, details=details, ocr_results=ocr_results)