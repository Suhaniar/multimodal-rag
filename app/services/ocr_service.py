import io
import base64
import re
import logging
import json
import ollama
from PIL import Image
from app.helpers.image_utils import preprocess_image
from app.helpers.text_utils import clean_text
from app.core.config import OLLAMA_OCR_MODEL, ENABLE_ENRICHMENT

logger = logging.getLogger("multimodal_rag")

def run_ocr(image: Image.Image) -> str:
    """Extract text from image using DeepSeek‑OCR."""
    try:
        image = preprocess_image(image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = ollama.chat(
            model=OLLAMA_OCR_MODEL,
            messages=[{
                'role': 'user',
                'content': (
                    "Extract all text exactly as it appears in this image. "
                    "If there is a table, preserve its structure using spaces or simple lines. "
                    "Do not add any commentary, explanation, or markdown formatting. "
                    "Do not hallucinate numbers. If something is unclear, skip it."
                ),
                'images': [img_base64]
            }]
        )
        raw_text = response['message']['content'].strip()
        cleaned = clean_text(raw_text)
        cleaned = re.sub(r'(Q\d)(\d+)', r'\1 \2', cleaned)
        return cleaned
    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        return ""   # ingestion will skip empty content

def enrich_image(image: Image.Image) -> dict:
    """Generate caption, objects, summary, and extract text."""
    if not ENABLE_ENRICHMENT:
        return {"caption": "", "objects": [], "summary": "", "extracted_text": ""}
    try:
        image = preprocess_image(image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = """Describe this image in JSON format with the following fields:
- "caption": a short phrase (max 10 words)
- "objects": list of main objects present (e.g., ["dog", "fire", "car"])
- "summary": a brief scene description (one sentence)
- "text": any text you see (if none, return empty string)

Output ONLY valid JSON, no extra text."""

        response = ollama.chat(
            model=OLLAMA_OCR_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_base64]
            }]
        )
        raw = response['message']['content'].strip()
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        enrichment = json.loads(raw)
        return {
            "caption": enrichment.get("caption", ""),
            "objects": enrichment.get("objects", []),
            "summary": enrichment.get("summary", ""),
            "extracted_text": enrichment.get("text", "")
        }
    except Exception as e:
        logger.error(f"Image enrichment failed: {e}", exc_info=True)
        return {"caption": "", "objects": [], "summary": "", "extracted_text": ""}

# ---------- Phase 2: Chart understanding ----------
def understand_chart(image: Image.Image) -> str:
    """Use vision model to interpret chart content."""
    try:
        image = preprocess_image(image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        prompt = (
            "This is a chart. Describe in detail: "
            "- the type of chart (bar, line, pie, etc.) "
            "- the axes and their labels "
            "- key trends, highest and lowest values "
            "- any notable data points. "
            "Output only the description, no extra text."
        )
        response = ollama.chat(
            model=OLLAMA_OCR_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_base64]
            }]
        )
        return response['message']['content'].strip()
    except Exception as e:
        logger.error(f"Chart understanding failed: {e}")
        return ""