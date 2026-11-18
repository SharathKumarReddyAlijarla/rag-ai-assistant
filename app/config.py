"""Application settings used by the whole manual assistant.

This file reads optional values from a .env file and turns them into Python
constants that the rest of the app can import. It decides where manuals,
models, and the vector database are stored, which embedding model is used,
which Ollama model should answer questions, and how large text chunks should
be during ingestion. It also contains small helper functions that resolve
project-relative paths and prefer a locally cached embedding model when one
has already been downloaded.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

def project_path(value: str | Path):
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


MANUALS_PATH = project_path(os.getenv("MANUALS_PATH", "data/manuals"))
VECTOR_DB_PATH = project_path(os.getenv("VECTOR_DB_PATH", "vector_store"))
MODELS_PATH = project_path(os.getenv("MODELS_PATH", "models"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "75"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))
TOP_K = int(os.getenv("TOP_K", "4"))

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))


def embedding_model_cache_path(model_name: str = EMBEDDING_MODEL_NAME):
    safe_name = model_name.replace("/", "__")
    return MODELS_PATH / safe_name


def embedding_model_location(model_name: str = EMBEDDING_MODEL_NAME):
    local_path = embedding_model_cache_path(model_name)
    return str(local_path) if local_path.exists() else model_name
