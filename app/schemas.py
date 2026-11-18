"""Shared data structures used across the manual assistant.

This file defines the simple objects that move through the app. ManualDocument
represents text extracted from a source manual. TextChunk represents a smaller
searchable piece of that manual. SearchResult pairs a chunk with its similarity
score from vector search. AskResponse defines the JSON shape returned by the
FastAPI /ask endpoint.
"""

from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel


@dataclass
class ManualDocument:
    text: str
    source: str
    doc_type: str
    page: int | None = None
    title: str | None = None


@dataclass
class TextChunk:
    id: str
    text: str
    source: str
    doc_type: str
    chunk_index: int
    page: int | None = None
    title: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float


class AskResponse(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
