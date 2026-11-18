"""Command-line script that builds the searchable manual index.

Run this file when manuals are added or changed. It loads manual files from the
manuals folder, cleans the extracted text, splits the text into chunks, creates
embeddings for each chunk, and saves the FAISS index plus chunk metadata into
the vector_store folder. The work is done in batches so large manual sets do
not need to be held in memory as one giant list.
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.chunking import chunk_documents
from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    MANUALS_PATH,
    VECTOR_DB_PATH,
)
from app.document_loader import SUPPORTED_EXTENSIONS, discover_manual_files, load_manual
from app.text_processing import clean_documents
from app.vector_store import save_vector_store_batches


def iter_chunk_batches(
    manuals_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
):
    files = discover_manual_files(manuals_path)
    if not files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported manuals found in {manuals_path}. Supported formats: {supported}"
        )

    batch = []

    for path in files:
        print(f"Loading {path.name}...")
        documents = load_manual(path)
        print(f"  Loaded {len(documents)} document sections.")

        cleaned_documents = clean_documents(documents)
        print(f"  Retained {len(cleaned_documents)} cleaned sections.")

        chunks = chunk_documents(cleaned_documents, chunk_size, chunk_overlap)
        print(f"  Created {len(chunks)} chunks.")

        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def run_ingestion(
    manuals_path: Path = MANUALS_PATH,
    vector_db_path: Path = VECTOR_DB_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = EMBEDDING_BATCH_SIZE,
):
    print(f"Reading manuals from {manuals_path}...")
    print(f"Chunk size: {chunk_size} chars, overlap: {chunk_overlap} chars.")
    print(f"Embedding with {EMBEDDING_MODEL_NAME} in batches of {batch_size}.")

    chunk_batches = iter_chunk_batches(
        manuals_path=manuals_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
    )
    chunk_count = save_vector_store_batches(
        chunk_batches,
        vector_db_path,
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=batch_size,
    )

    print(f"Saved FAISS index and SQLite chunk store to {vector_db_path}.")
    print(f"Indexed {chunk_count} chunks.")
    return chunk_count


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest manuals into the local FAISS vector store.")
    parser.add_argument("--manuals-path", type=Path, default=MANUALS_PATH)
    parser.add_argument("--vector-db-path", type=Path, default=VECTOR_DB_PATH)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_ingestion(
            manuals_path=args.manuals_path,
            vector_db_path=args.vector_db_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from None
