"""Download and cache models needed for offline use.

This helper script prepares the machine before running the assistant offline.
It downloads the sentence-transformers embedding model and saves it under the
local models folder. It can also ask Ollama to pull the configured chat model,
unless --skip-ollama is used. After this setup, the app can use local cached
models instead of downloading them during normal operation.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, OLLAMA_MODEL, embedding_model_cache_path


def parse_args():
    parser = argparse.ArgumentParser(description="Download local models required for offline use.")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--skip-ollama", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Caching sentence-transformers model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    target_path = embedding_model_cache_path(args.embedding_model)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target_path))
    print(f"Saved embedding model to {target_path}")

    if not args.skip_ollama:
        print(f"Pulling Ollama model: {args.ollama_model}")
        subprocess.run(["ollama", "pull", args.ollama_model], check=True)

    print("Model setup complete.")


if __name__ == "__main__":
    main()
