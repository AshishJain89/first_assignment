import io
import PyPDF2
from typing import List

def pdf_to_text(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    texts = []
    for p in range(len(reader.pages)):
        try:
            texts.append(reader.pages[p].extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
