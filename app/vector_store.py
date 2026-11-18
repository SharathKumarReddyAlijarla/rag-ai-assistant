"""Save, load, and search the local vector database.

This file keeps the app free and local while making storage faster and more
scalable than one large JSON file. FAISS stores the numeric vectors used for
similarity search. SQLite stores the matching chunk text and metadata, so the
app can fetch only the few chunks returned by search instead of loading every
chunk into memory on each question. Ingestion writes chunks in batches, which
is safer for large manual collections.
"""

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import faiss

from app.config import EMBEDDING_MODEL_NAME
from app.embeddings import embed_query, embed_texts
from app.schemas import SearchResult, TextChunk

INDEX_FILE = "index.faiss"
CHUNKS_DB_FILE = "chunks.sqlite3"
DEFAULT_BATCH_SIZE = 256
HNSW_NEIGHBORS = 32
HNSW_EF_CONSTRUCTION = 100
HNSW_EF_SEARCH = 64


def index_path(vector_db_path: Path):
    return vector_db_path / INDEX_FILE


def chunks_db_path(vector_db_path: Path):
    return vector_db_path / CHUNKS_DB_FILE


def vector_store_exists(vector_db_path: Path):
    return index_path(vector_db_path).exists() and chunks_db_path(vector_db_path).exists()


def connect_chunks_db(vector_db_path: Path):
    conn = sqlite3.connect(chunks_db_path(vector_db_path))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_chunks_db(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            vector_id INTEGER PRIMARY KEY,
            id TEXT NOT NULL UNIQUE,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page INTEGER,
            title TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()


def reset_vector_store(vector_db_path: Path):
    vector_db_path.mkdir(parents=True, exist_ok=True)
    for path in (index_path(vector_db_path), chunks_db_path(vector_db_path)):
        if path.exists():
            path.unlink()


def create_search_index(dimension: int):
    index = faiss.IndexHNSWFlat(dimension, HNSW_NEIGHBORS, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    return index


def batched(items: Iterable[TextChunk], batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def insert_chunks(conn: sqlite3.Connection, chunks: list[TextChunk], start_vector_id: int):
    rows = [
        (
            start_vector_id + offset,
            chunk.id,
            chunk.text,
            chunk.source,
            chunk.doc_type,
            chunk.chunk_index,
            chunk.page,
            chunk.title,
        )
        for offset, chunk in enumerate(chunks)
    ]
    conn.executemany(
        """
        INSERT INTO chunks (
            vector_id, id, text, source, doc_type, chunk_index, page, title
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def save_metadata(
    conn: sqlite3.Connection,
    model_name: str,
    chunk_count: int,
):
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model_name,
        "chunk_count": str(chunk_count),
        "index_type": "faiss.IndexHNSWFlat",
    }
    conn.executemany(
        """
        INSERT INTO metadata (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        metadata.items(),
    )
    conn.commit()


def save_vector_store(
    chunks: list[TextChunk],
    vector_db_path: Path,
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    return save_vector_store_batches(
        batched(chunks, batch_size),
        vector_db_path,
        model_name=model_name,
        batch_size=batch_size,
    )


def save_vector_store_batches(
    chunk_batches: Iterable[list[TextChunk]],
    vector_db_path: Path,
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    reset_vector_store(vector_db_path)
    conn = connect_chunks_db(vector_db_path)
    initialize_chunks_db(conn)

    index = None
    total_chunks = 0

    try:
        for source_batch in chunk_batches:
            for chunks in batched(source_batch, batch_size):
                texts = [chunk.text for chunk in chunks]
                vectors = embed_texts(texts, model_name=model_name, show_progress=False)

                if vectors.ndim != 2 or vectors.shape[0] == 0:
                    continue

                if index is None:
                    index = create_search_index(vectors.shape[1])

                index.add(vectors)
                insert_chunks(conn, chunks, total_chunks)
                total_chunks += len(chunks)
                print(f"Embedded and stored {total_chunks} chunks...")

        if index is None or total_chunks == 0:
            raise RuntimeError("No chunks were produced from the manuals.")

        save_metadata(conn, model_name, total_chunks)
        faiss.write_index(index, str(index_path(vector_db_path)))
        return total_chunks
    finally:
        conn.close()


def row_to_chunk(row: sqlite3.Row):
    return TextChunk(
        id=row["id"],
        text=row["text"],
        source=row["source"],
        doc_type=row["doc_type"],
        chunk_index=row["chunk_index"],
        page=row["page"],
        title=row["title"],
    )


def chunk_count(conn: sqlite3.Connection):
    row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
    return int(row["count"])


def fetch_chunks_by_vector_id(conn: sqlite3.Connection, vector_ids: list[int]):
    if not vector_ids:
        return {}

    placeholders = ",".join("?" for _ in vector_ids)
    rows = conn.execute(
        f"""
        SELECT vector_id, id, text, source, doc_type, chunk_index, page, title
        FROM chunks
        WHERE vector_id IN ({placeholders})
        """,
        vector_ids,
    ).fetchall()
    return {int(row["vector_id"]): row_to_chunk(row) for row in rows}


def load_search_index(vector_db_path: Path):
    if not vector_store_exists(vector_db_path):
        raise FileNotFoundError(
            f"Vector store not found in {vector_db_path}. Run python app/ingest.py first."
        )

    index = faiss.read_index(str(index_path(vector_db_path)))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = HNSW_EF_SEARCH
    return index


def search_vector_store(
    query: str,
    vector_db_path: Path,
    top_k: int,
    model_name: str = EMBEDDING_MODEL_NAME,
):
    index = load_search_index(vector_db_path)

    if index.ntotal == 0:
        return []

    conn = connect_chunks_db(vector_db_path)
    try:
        stored_chunks = chunk_count(conn)
        if index.ntotal != stored_chunks:
            raise RuntimeError(
                f"Vector index/chunk metadata mismatch: index={index.ntotal}, chunks={stored_chunks}"
            )

        query_vector = embed_query(query, model_name=model_name)
        scores, indices = index.search(query_vector, min(top_k, index.ntotal))

        vector_ids = [int(index_position) for index_position in indices[0] if index_position >= 0]
        chunks_by_vector_id = fetch_chunks_by_vector_id(conn, vector_ids)
    finally:
        conn.close()

    results = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue

        chunk = chunks_by_vector_id.get(int(index_position))
        if chunk is None:
            continue

        results.append(SearchResult(chunk=chunk, score=float(score)))

    return results
