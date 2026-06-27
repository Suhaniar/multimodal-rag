import re
from typing import List, Tuple, Any, Dict
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    return " ".join((text or "").split())

def rows_to_text(rows: list[list[Any]]) -> str:
    lines = []
    for row in rows:
        cleaned = [clean_text(str(cell or "")) for cell in row]
        if any(cleaned):
            lines.append(" | ".join(cleaned))
    return "\n".join(lines)

def document_key(doc: Document) -> tuple[str, str, str, str]:
    return (
        doc.metadata.get("source", ""),
        str(doc.metadata.get("page", "")),
        doc.metadata.get("modality", ""),
        doc.page_content.lower(),
    )

def deduplicate_documents(docs: list[Document]) -> list[Document]:
    seen = set()
    result = []
    for doc in docs:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result

def tokenize_bm25(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

# ---------- Section / Heading detection ----------
def detect_headings(text: str) -> List[Tuple[int, str]]:
    """
    Return list of (line_index, heading_text) for lines that are likely headings.
    Uses heuristics: all caps, numbered, or ending with colon.
    """
    lines = text.split('\n')
    headings = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if (line.isupper() and len(line) < 60) or \
           re.match(r'^(\d+\.?\s*[A-Z])', line) or \
           (line.endswith(':') and len(line) < 50):
            headings.append((i, line))
    return headings

def assign_sections_to_lines(text: str, headings: List[Tuple[int, str]]) -> List[str]:
    """
    Return a list of section labels per line.
    """
    lines = text.split('\n')
    if not headings:
        return [""] * len(lines)
    section_labels = []
    current_section = ""
    heading_dict = {idx: heading for idx, heading in headings}
    for i in range(len(lines)):
        if i in heading_dict:
            current_section = heading_dict[i]
        section_labels.append(current_section)
    return section_labels

def extract_headings_from_pdf(pdf) -> List[Tuple[int, str]]:
    """
    Extract headings from all pages of a PyMuPDF document.
    Returns list of (page_num, heading_text).
    """
    import pymupdf as fitz
    headings = []
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text")
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if (line.isupper() and len(line) < 80) or re.match(r'^(\d+\.?\s*[A-Z])', line):
                headings.append((page_num + 1, line))
    return headings

def infer_heading_level(heading_text: str) -> int:
    """
    Infer heading level from heading text.
    - Starts with '#' -> count # (for markdown)
    - Starts with number pattern like '1.' or '1.1.' etc.
    - All caps: treat as H1
    - Ending with colon: treat as H2
    """
    heading = heading_text.strip()
    if heading.startswith('#'):
        level = heading.count('#')
        return min(level, 4)
    if re.match(r'^(\d+\.?\s*[A-Z])', heading):
        dots = heading.split('.')
        parts = [p for p in dots if p.strip().isdigit()]
        level = len(parts) if parts else 1
        return min(level, 4)
    if heading.isupper() and len(heading) < 60:
        return 1
    if heading.endswith(':') and len(heading) < 50:
        return 2
    return 3  # default

def build_section_tree(headings: List[Tuple[int, str]]) -> Dict[int, str]:
    """
    Given list of (line_index, heading_text), build a mapping from line_index
    to the full section path (e.g., "Chapter 1 > Introduction") for every line.
    Lines before the first heading get an empty string.
    """
    if not headings:
        return {}

    path = []
    section_map = {}
    # First, assign sections to heading lines
    for line_idx, heading in headings:
        level = infer_heading_level(heading)
        # Adjust path to current level
        while len(path) >= level:
            path.pop()
        path.append(heading)
        section_map[line_idx] = " > ".join(path)

    return section_map

def build_global_section_paths(headings_per_page: Dict[int, List[Tuple[int, str]]]) -> Dict[int, str]:
    """
    Given a dict mapping page number to list of (line_index, heading), build a section path for each page.
    Returns dict {page_num: section_path_string}.
    """
    path = []
    section_per_page = {}
    # Flatten headings with page info
    all_headings = []
    for page, headings in headings_per_page.items():
        for idx, heading in headings:
            all_headings.append((page, idx, heading))
    # Process in page order
    current_section = ""
    for page, idx, heading in all_headings:
        level = infer_heading_level(heading)
        while len(path) >= level:
            path.pop()
        path.append(heading)
        current_section = " > ".join(path)
        # Assign to this page (and all subsequent pages until next heading)
        section_per_page[page] = current_section
    # For pages without any heading, carry previous section
    prev_section = ""
    for page in sorted(headings_per_page.keys()):
        if page in section_per_page:
            prev_section = section_per_page[page]
        else:
            section_per_page[page] = prev_section
    return section_per_page