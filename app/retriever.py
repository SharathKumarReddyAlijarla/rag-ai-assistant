"""Retrieve the most relevant manual chunks for a question.

This file is a small wrapper around the vector store search. It receives the
user's question, keeps the requested number of results within a safe range, and
searches the saved FAISS database using the configured embedding model. The
returned chunks are later passed to the LLM as context for the final answer.
"""

from app.config import EMBEDDING_MODEL_NAME, TOP_K, VECTOR_DB_PATH
from app.vector_store import search_vector_store


def retrieve_context(query: str, top_k: int = TOP_K):
    top_k = max(1, min(top_k, 10))
    return search_vector_store(
        query=query,
        vector_db_path=VECTOR_DB_PATH,
        top_k=top_k,
        model_name=EMBEDDING_MODEL_NAME,
    )
