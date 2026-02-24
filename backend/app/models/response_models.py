from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str


class UploadResponse(BaseModel):
    video_id: str
    title: str
    collection_name: str
    chunk_count: int
    transcript_path: str


class SourceChunk(BaseModel):
    text: str
    metadata: dict
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    tokens_used: int


class GenericResponse(BaseModel):
    message: str
