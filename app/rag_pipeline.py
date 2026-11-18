"""Run the full retrieval-augmented question answering flow.

This file connects the main RAG steps together. It receives a user question,
retrieves the best matching manual chunks, sends only the chunk text to the LLM,
and returns the generated answer with a simple confidence value. If no useful
context is found, or the LLM says the answer is not in the manual, it returns
the standard "Not found in manual" response.
"""

from app.config import TOP_K
from app.llm import NOT_FOUND, build_prompt, call_ollama
from app.retriever import retrieve_context


def confidence_from_score(score: float, answer: str):
    if answer == NOT_FOUND:
        return "low"
    if score >= 0.55:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def format_context_block(result):
    return result.chunk.text


def ask_question(query: str, top_k: int = TOP_K):
    query = query.strip()
    if not query:
        raise ValueError("Question cannot be empty.")

    results = retrieve_context(query, top_k=top_k)
    if not results:
        return {
            "answer": NOT_FOUND,
            "confidence": "low",
        }

    context_blocks = [
        format_context_block(result)
        for result in results
    ]
    prompt = build_prompt(query, context_blocks)
    answer = call_ollama(prompt)

    if answer == NOT_FOUND:
        return {
            "answer": NOT_FOUND,
            "confidence": "low",
        }

    best_score = results[0].score

    return {
        "answer": answer,
        "confidence": confidence_from_score(best_score, answer),
    }
