"""Load manual files from disk and turn them into document objects.

This file is responsible for reading the supported manual formats: PDF, DOCX,
HTML, TXT, and Markdown. It searches the manuals folder, picks only supported
files, extracts readable text from each file, and stores that text with basic
metadata such as file name, document type, page number, and title. Later steps
clean and split these document objects before they are embedded and searched.
"""

from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.schemas import ManualDocument

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}


def discover_manual_files(manuals_path: Path):
    manuals_path.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in manuals_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_manuals(manuals_path: Path):
    files = discover_manual_files(manuals_path)
    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported manuals found in {manuals_path}. Supported formats: {supported}"
        )

    documents = []
    for path in files:
        documents.extend(load_manual(path))

    return documents


def load_manual(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in {".html", ".htm"}:
        return load_html(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)

    return []


def load_pdf(path: Path):
    reader = PdfReader(str(path))
    documents = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                ManualDocument(
                    text=text,
                    source=path.name,
                    doc_type="pdf",
                    page=page_index,
                    title=path.stem,
                )
            )

    return documents


def load_docx(path: Path):
    doc = DocxDocument(str(path))
    blocks = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    text = "\n".join(blocks)
    return [
        ManualDocument(
            text=text,
            source=path.name,
            doc_type="docx",
            title=path.stem,
        )
    ] if text.strip() else []


def load_html(path: Path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
    text = soup.get_text(separator="\n")

    return [
        ManualDocument(
            text=text,
            source=path.name,
            doc_type="html",
            title=title,
        )
    ] if text.strip() else []


def load_text(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        ManualDocument(
            text=text,
            source=path.name,
            doc_type=path.suffix.lower().lstrip("."),
            title=path.stem,
        )
    ] if text.strip() else []
