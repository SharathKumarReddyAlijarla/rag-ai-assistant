"""FastAPI backend for the manual assistant.

This file exposes the app as HTTP endpoints. The /health endpoint reports
whether the vector store is ready and which Ollama settings are being used.
The /ask endpoint receives a question from the UI or another client, runs the
RAG pipeline, and returns the generated answer. It also converts common errors
into clear HTTP responses so the UI can show useful messages.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, VECTOR_DB_PATH
from app.rag_pipeline import ask_question
from app.schemas import AskResponse
from app.vector_store import vector_store_exists

app = FastAPI(title="Offline Manual AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store_ready": vector_store_exists(VECTOR_DB_PATH),
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
    }


@app.get("/ask", response_model=AskResponse)
def ask(q: str = Query(..., min_length=1), top_k: int = Query(4, ge=1, le=10)):
    try:
        return ask_question(q, top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        status_code = 503 if "Ollama" in message else 500
        raise HTTPException(status_code=status_code, detail=message) from exc
