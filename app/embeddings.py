"""Create numeric embeddings for manual chunks and user questions.

Embeddings are vectors that represent the meaning of text. This file loads the
sentence-transformers model, caches it so it is not reloaded for every request,
and converts both manual chunks and user questions into normalized float32
vectors. The vector store uses these vectors to find the manual chunks that are
most similar to the user's question.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, embedding_model_location


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    return SentenceTransformer(embedding_model_location(model_name))


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME, show_progress: bool = False):
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str, model_name: str = EMBEDDING_MODEL_NAME):
    model = get_embedding_model(model_name)
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embedding, dtype="float32")
