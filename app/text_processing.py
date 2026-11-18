"""Clean extracted manual text before it is split into chunks.

Manual text often contains repeated headers, footers, page numbers, strange
line breaks, extra spaces, and broken words from PDF extraction. This file
removes that noise so the embedding model and LLM receive cleaner content.
It also provides a simple sentence splitter that keeps chunking closer to
natural sentences instead of cutting text in the middle of an idea.
"""

import re
from collections import Counter

from app.schemas import ManualDocument

PAGE_LINE_RE = re.compile(r"^(page\s*)?\d+(\s+of\s+\d+)?$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"[ \t]+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_line(line: str):
    return WHITESPACE_RE.sub(" ", line.strip()).lower()


def detect_repeated_noise_lines(documents: list[ManualDocument]):
    line_counts = Counter()
    total_documents = len(documents)

    if total_documents < 3:
        return set()

    for document in documents:
        unique_lines = {
            normalize_line(line)
            for line in document.text.splitlines()
            if 4 <= len(normalize_line(line)) <= 120
        }
        line_counts.update(unique_lines)

    min_count = max(3, int(total_documents * 0.35))
    return {
        line
        for line, count in line_counts.items()
        if count >= min_count and not line.endswith((".", ":", "?"))
    }


def clean_text(text: str, noise_lines: set[str] | None = None):
    noise_lines = noise_lines or set()
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    previous_blank = False

    for raw_line in text.split("\n"):
        line = WHITESPACE_RE.sub(" ", raw_line.strip())
        normalized = normalize_line(line)

        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        if PAGE_LINE_RE.match(line) or normalized in noise_lines:
            continue

        cleaned_lines.append(line)
        previous_blank = False

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_documents(documents: list[ManualDocument]):
    noise_lines = detect_repeated_noise_lines(documents)
    cleaned = []

    for document in documents:
        text = clean_text(document.text, noise_lines)
        if text:
            cleaned.append(
                ManualDocument(
                    text=text,
                    source=document.source,
                    doc_type=document.doc_type,
                    page=document.page,
                    title=document.title,
                )
            )

    return cleaned


def split_sentences(text: str):
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    segments = []

    for paragraph in paragraphs:
        sentences = SENTENCE_BOUNDARY_RE.split(paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                segments.append(sentence)

    return segments
