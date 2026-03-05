import os
import re
from typing import List, Dict, Any, Tuple, Optional

from pypdf import PdfReader
from docx import Document

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

SECTION_PATTERNS = [
    re.compile(r"^\s*第\s*[0-9一二三四五六七八九十百千]+\s*章\b"),
    re.compile(r"^\s*[壹貳參肆伍陸柒捌玖拾]+、\s*.+"),
    re.compile(r"^\s*[一二三四五六七八九十]+、\s*.+"),
    re.compile(r"^\s*（[一二三四五六七八九十]+）\s*.+"),
    re.compile(r"^\s*\d+(\.\d+)*\s+.+"),
]

def normalize_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_section_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) > 60:
        return False
    return any(p.search(s) for p in SECTION_PATTERNS)

def build_doc_id(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", filename)

def extract_docx_text(path: str) -> str:
    doc = Document(path)
    paras = []
    for p in doc.paragraphs:
        if p.text:
            paras.append(p.text)
    return "\n".join(paras)

def pdf_page_text(reader: PdfReader, page_index: int) -> str:
    try:
        return reader.pages[page_index].extract_text() or ""
    except Exception:
        return ""

def ocr_pdf_page(path: str, page_no_1based: int, dpi: int, lang: str) -> str:
    images = convert_from_path(
        path,
        dpi=dpi,
        first_page=page_no_1based,
        last_page=page_no_1based,
        fmt="png",
        thread_count=1,
    )
    if not images:
        return ""
    img: Image.Image = images[0].convert("L")
    return pytesseract.image_to_string(img, lang=lang) or ""

def should_ocr_pdf(page_texts: List[str], min_chars_per_page: int, page_ratio: float, total_min_chars: int) -> bool:
    if not page_texts:
        return True
    total = sum(len((t or "").strip()) for t in page_texts)
    if total < total_min_chars:
        return True
    low = sum(1 for t in page_texts if len((t or "").strip()) < min_chars_per_page)
    return (low / max(1, len(page_texts))) >= page_ratio

def chunk_with_sections(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    text = normalize_text(text)
    if not text:
        return []

    lines = text.splitlines()
    current_section = None

    annotated: List[Tuple[str, Optional[str]]] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        if is_section_heading(s):
            current_section = s
        annotated.append((s, current_section))

    flat = "\n".join([a[0] for a in annotated])
    if not flat:
        return []

    positions: List[Tuple[int, Optional[str]]] = []
    cur = 0
    for line, sec in annotated:
        positions.append((cur, sec))
        cur += len(line) + 1

    def section_at(pos: int) -> Optional[str]:
        last = None
        for start, sec in positions:
            if start > pos:
                break
            if sec:
                last = sec
        return last

    n = len(flat)
    out: List[Dict[str, Any]] = []
    start = 0
    cid = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk_txt = flat[start:end].strip()
        if chunk_txt:
            out.append({
                "chunk_id": cid,
                "text": chunk_txt,
                "section_path": section_at(start),
            })
            cid += 1
        if end == n:
            break
        start = max(0, end - overlap)
    return out

def parse_pdf(path: str, filename: str, chunk_size: int, overlap: int) -> Dict[str, Any]:
    reader = PdfReader(path)
    num_pages = len(reader.pages)

    ocr_enabled = os.getenv("OCR_ENABLED", "true").lower() == "true"
    ocr_lang = os.getenv("OCR_LANG", "chi_tra+eng")
    ocr_dpi = int(os.getenv("OCR_DPI", "220"))
    min_chars = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "30"))
    ratio = float(os.getenv("OCR_PAGE_RATIO", "0.6"))
    total_min = int(os.getenv("OCR_TOTAL_MIN_CHARS", "300"))
    max_pages = int(os.getenv("OCR_MAX_PAGES", "60"))

    pages_to_read = min(num_pages, max_pages)

    page_texts: List[str] = [pdf_page_text(reader, i) for i in range(pages_to_read)]
    need_ocr = ocr_enabled and should_ocr_pdf(page_texts, min_chars, ratio, total_min)

    all_chunks: List[Dict[str, Any]] = []
    for i in range(pages_to_read):
        page_no = i + 1
        text = page_texts[i]

        if need_ocr and len((text or "").strip()) < min_chars:
            text = ocr_pdf_page(path, page_no_1based=page_no, dpi=ocr_dpi, lang=ocr_lang)

        chunks = chunk_with_sections(text, chunk_size, overlap)
        for c in chunks:
            c["page"] = page_no
        all_chunks.extend(chunks)

    doc_id = build_doc_id(filename)
    return {"doc_id": doc_id, "filename": filename, "chunks": all_chunks, "meta": {"need_ocr": need_ocr}}

def parse_file(path: str, filename: str, chunk_size: int, overlap: int) -> Dict[str, Any]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return parse_pdf(path, filename, chunk_size, overlap)
    if ext == ".docx":
        text = extract_docx_text(path)
        chunks = chunk_with_sections(text, chunk_size, overlap)
        for c in chunks:
            c["page"] = None
        doc_id = build_doc_id(filename)
        return {"doc_id": doc_id, "filename": filename, "chunks": chunks, "meta": {"need_ocr": False}}
    raise ValueError(f"Unsupported file type: {ext}")