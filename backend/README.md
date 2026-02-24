# Backend (FastAPI + Milvus Lite + RAG)

## Features

- YouTube ingestion pipeline: URL -> video -> audio -> transcript -> chunks -> embeddings -> Milvus Lite
- Milvus Lite local file mode via `pymilvus` (`connections.connect(uri='./milvus.db')`)
- HNSW index with cosine similarity
- Per-video collection strategy (configurable)
- RAG chat endpoint with strict context-only prompt
- Optional SSE streaming endpoint
- Collection delete and rebuild endpoints

## Project Structure

```text
backend/
  app/
    main.py
    api/
      upload.py
      chat.py
    services/
      youtube_service.py
      audio_service.py
      transcription_service.py
      embedding_service.py
      milvus_service.py
      rag_service.py
      pipeline_service.py
    core/
      config.py
      logging.py
    models/
      request_models.py
      response_models.py
    utils/
      dependencies.py
    vectorstore/
      langchain_milvus_store.py
```

## Prerequisites

- Python 3.11+
- FFmpeg available in PATH (required by `moviepy`)
- OpenAI-compatible API key

## Setup

1. Create venv and install deps:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure env:

```bash
cp .env.example .env
```

3. Start API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open docs:

- `http://localhost:8000/docs`

## Milvus Lite Local Mode

Milvus Lite is started implicitly when backend initializes:

```python
from pymilvus import connections
connections.connect(uri='./milvus.db')
```

The `milvus.db` file is created in backend working directory (or the configured `APP_MILVUS_URI`).

## API Examples

### Health

```bash
curl http://localhost:8000/health
```

### Upload and Index a Video

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H 'Content-Type: application/json' \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the video about?","video_id":"dQw4w9WgXcQ"}'
```

### Stream Chat (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"Summarize in 3 bullets","video_id":"dQw4w9WgXcQ"}'
```

### Delete Collection

```bash
curl -X DELETE http://localhost:8000/api/v1/upload/collection \
  -H 'Content-Type: application/json' \
  -d '{"video_id":"dQw4w9WgXcQ"}'
```

### Rebuild Collection

```bash
curl -X POST http://localhost:8000/api/v1/upload/rebuild \
  -H 'Content-Type: application/json' \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

## Docker

Build and run backend only:

```bash
cd backend
docker build -t video-bot-backend .
docker run --env-file .env -p 8000:8000 video-bot-backend
```
