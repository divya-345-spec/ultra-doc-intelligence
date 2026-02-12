import os
from pypdf import PdfReader
from docx import Document


def parse_pdf(file_path):
    """Extract text page-wise from a PDF."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def parse_docx(file_path):
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if full_text.strip():
        return [{"page": 1, "text": full_text}]
    return []


def parse_txt(file_path):
    """Extract text from a TXT file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if text:
        return [{"page": 1, "text": text}]
    return []


def parse_documents(folder_path):
    """
    Read all supported files (PDF, DOCX, TXT) from a folder
    and extract text page-wise.
    """
    documents = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        lower = filename.lower()

        if lower.endswith(".pdf"):
            documents[filename] = parse_pdf(file_path)
        elif lower.endswith(".docx"):
            documents[filename] = parse_docx(file_path)
        elif lower.endswith(".txt"):
            documents[filename] = parse_txt(file_path)

    return documents


# Keep backward compatibility
parse_all_pdfs = parse_documents
