"""Split cleaned manual text into searchable chunks.

The assistant cannot send an entire manual to the embedding model or LLM at
once, so this file breaks each cleaned document into smaller text pieces. It
tries to keep sentences together, splits very long sentences when needed, and
adds a small overlap between chunks so important context is not lost at chunk
boundaries. Each chunk also gets a stable id and metadata so it can be saved
and found again in the vector store.
"""

import re
from hashlib import sha1

from app.schemas import ManualDocument, TextChunk
from app.text_processing import split_sentences


def split_long_segment(segment: str, max_size: int):
    words = segment.split()
    pieces = []
    current = []
    current_size = 0

    for word in words:
        word_size = len(word) + 1
        if current and current_size + word_size > max_size:
            pieces.append(" ".join(current))
            current = [word]
            current_size = word_size
        else:
            current.append(word)
            current_size += word_size

    if current:
        pieces.append(" ".join(current))

    return pieces


def overlap_suffix(text: str, overlap_size: int):
    if overlap_size <= 0 or len(text) <= overlap_size:
        return text if overlap_size > 0 else ""

    words = text.split()
    selected = []
    size = 0

    for word in reversed(words):
        size += len(word) + 1
        selected.append(word)
        if size >= overlap_size:
            break

    return " ".join(reversed(selected))


def make_chunk_id(document: ManualDocument, chunk_index: int, text: str):
    key = f"{document.source}:{document.page}:{chunk_index}:{text[:120]}"
    digest = sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{document.source}:{document.page or 'doc'}:{chunk_index}:{digest}"


def chunk_document(document: ManualDocument, chunk_size: int, overlap_size: int):
    raw_segments = split_sentences(document.text)
    segments = []

    for segment in raw_segments:
        if len(segment) > chunk_size:
            segments.extend(split_long_segment(segment, chunk_size))
        else:
            segments.append(segment)

    chunks = []
    current = ""

    for segment in segments:
        candidate = f"{current} {segment}".strip() if current else segment

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = f"{overlap_suffix(current, overlap_size)} {segment}".strip()
        else:
            chunks.append(segment)
            current = ""

    if current:
        chunks.append(current)

    text_chunks = []
    for index, text in enumerate(chunks):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        text_chunks.append(
            TextChunk(
                id=make_chunk_id(document, index, text),
                text=text,
                source=document.source,
                doc_type=document.doc_type,
                page=document.page,
                title=document.title,
                chunk_index=index,
            )
        )

    return text_chunks


def chunk_documents(documents: list[ManualDocument], chunk_size: int, overlap_size: int):
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, overlap_size))

    return chunks
