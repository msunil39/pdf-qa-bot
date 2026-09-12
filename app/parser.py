import fitz
import re
import logging

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def extract_text(pdf_bytes: bytes) -> str:

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n".join(pages)
        return clean_text(text)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


def clean_text(text: str) -> str:

    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_text(text: str,
               chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            boundary = text.rfind(".", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_id":   chunk_id,
                "text":       chunk,
                "start":      start,
                "end":        end,
                "char_count": len(chunk),
            })
            chunk_id += 1

        start = end - overlap

    logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
    return chunks
