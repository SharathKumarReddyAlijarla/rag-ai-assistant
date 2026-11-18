"""Build prompts and call Ollama to generate final answers.

This file takes the retrieved manual text and the user's question, then builds
a strict prompt for the local LLM. The prompt tells the model to answer only
from the manual context, avoid source/debug text, and return a clear short
answer. The file also sends the prompt to Ollama and normalizes empty or
not-found responses into one consistent message.
"""

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, REQUEST_TIMEOUT_SECONDS

NOT_FOUND = "Not found in manual"


def build_prompt(question: str, context_blocks: list[str]):
    context = "\n\n".join(context_blocks).strip()
    return f"""You are a technical manual assistant.

Rules:
- Answer ONLY using the provided manual context.
- If the answer is not in the manual context, respond exactly: {NOT_FOUND}
- Do not use outside knowledge.
- Write a direct, polished answer to the user's question.
- Do not mention chunk IDs, scores, source labels, or phrases like "According to [1]".
- Do not quote raw manual fragments unless the user asks for an exact quote.
- Turn procedural fragments into clear user-facing instructions.
- Answer directly; do not start with filler such as "Based on the manual", "According to the context", or "follow these steps".
- Output format: one paragraph, 2-4 sentences.
- Do not use bullets, numbered lists, tables, or markdown unless the user explicitly asks for that format.
- A question that starts with "how" is not by itself a request for a numbered list.

Manual context:
{context}

Question:
{question}

Answer:"""


def call_ollama(prompt: str, model: str = OLLAMA_MODEL):
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.2,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Start Ollama and pull model '{model}'."
        ) from exc

    data = response.json()
    answer = (data.get("response") or "").strip()
    return normalize_answer(answer)


def normalize_answer(answer: str):
    if not answer:
        return NOT_FOUND

    normalized = answer.strip().strip('"').strip("'").strip()
    if normalized.lower().startswith(NOT_FOUND.lower()):
        return NOT_FOUND

    return normalized
