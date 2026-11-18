# Offline Manual AI Assistant

Local Retrieval-Augmented Generation system for technical manuals.

## Project Structure

```text
app/
  config.py           Runtime settings from .env
  document_loader.py  PDF, DOCX, HTML, text loading
  text_processing.py  Cleaning and noise removal
  chunking.py         Sentence-aware chunking
  embeddings.py       sentence-transformers embeddings
  vector_store.py     FAISS vector search plus SQLite chunk storage
  retriever.py        Retrieval wrapper
  llm.py              Ollama generation
  rag_pipeline.py     Retrieval plus grounded answer generation
  main.py             FastAPI backend
  ingest.py           Manual ingestion script
ui/
  app.py              Streamlit UI
scripts/
  download_models.py  Optional setup script for offline model caching
data/manuals/         Put PDF, DOCX, HTML, TXT, or MD manuals here
models/               Local sentence-transformers model cache
vector_store/         Local FAISS index and SQLite chunk metadata
```

## Setup

1. Install Ollama.
2. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Download models once while online:

```powershell
python scripts/download_models.py --ollama-model llama3
```

You can use `mistral` or `phi3` instead:

```powershell
python scripts/download_models.py --ollama-model phi3
```

After this setup, the assistant can run offline as long as Ollama and the cached embedding model are available locally.

## Ingest Manuals

Place manuals in:

```text
data/manuals/
```

Then run:

```powershell
python app/ingest.py
```

This creates:

```text
vector_store/index.faiss
vector_store/chunks.sqlite3
```

## Run Backend

```powershell
uvicorn app.main:app --reload
```

API:

```text
GET http://localhost:8000/ask?q=<question>
GET http://localhost:8000/health
```

Response shape:

```json
{
  "answer": "...",
  "confidence": "high"
}
```

If the answer is not in the manuals, the answer is exactly:

```text
Not found in manual
```

## Run UI

```powershell
streamlit run ui/app.py
```

## Configuration

Optional `.env` values:

```text
MANUALS_PATH=data/manuals
VECTOR_DB_PATH=vector_store
MODELS_PATH=models
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
CHUNK_SIZE=500
CHUNK_OVERLAP=75
EMBEDDING_BATCH_SIZE=256
TOP_K=4
REQUEST_TIMEOUT_SECONDS=120
```
