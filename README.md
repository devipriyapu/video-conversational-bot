# Video Conversational Bot (Angular + FastAPI + Milvus Lite)

Full-stack application that indexes YouTube videos and supports RAG chat over transcript chunks.

<img width="1670" height="826" alt="Screenshot 2026-02-24 at 9 38 58 AM" src="https://github.com/user-attachments/assets/3c2a9068-8f2d-4de7-b17c-e3fa7d6cf2d8" />


## Monorepo Layout

-   `backend/`: FastAPI, pipeline services, Milvus Lite, RAG orchestration
-   `frontend/`: Angular + Material chat/upload UI
-   `docker-compose.yml`: optional local orchestration

## End-to-End Flow

1.  User submits YouTube URL in Angular UI.
2.  Backend downloads video (`yt-dlp`).
3.  Backend extracts MP3 audio (`moviepy`).
4.  Backend transcribes audio (`faster-whisper`).
5.  Transcript is chunked (`RecursiveCharacterTextSplitter`).
6.  Chunks embedded (OpenAI-compatible embeddings API).
7.  Embeddings and chunk payload stored in Milvus Lite.
8.  User asks chatbot question.
9.  Backend retrieves top-K chunks via cosine similarity and answers with strict RAG prompt.

## Quick Start

### 1. Backend

```bash
cd backendcp .env.example .envpython -m venv .venvsource .venv/bin/activatepip install -r requirements.txtuvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontendnpm installnpm start
```

Open `http://localhost:4200`.

## Environment Notes

-   Backend env file: `backend/.env`
-   Frontend API base URL: `frontend/src/app/environments/environment.ts`
-   Default backend API prefix: `/api/v1`

## Milvus Lite Setup

No standalone Milvus server is required.Milvus Lite runs in file mode with:

```python
connections.connect(uri='./milvus.db')
```

Database file location controlled by `APP_MILVUS_URI` in `backend/.env`.

## Curl Smoke Tests

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/api/v1/upload   -H 'Content-Type: application/json'   -d '{"youtube_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

```bash
curl -X POST http://localhost:8000/api/v1/chat   -H 'Content-Type: application/json'   -d '{"question":"Give me a summary","video_id":"dQw4w9WgXcQ"}'
```

## Docker Compose (Optional)

```bash
docker compose up --build
```
