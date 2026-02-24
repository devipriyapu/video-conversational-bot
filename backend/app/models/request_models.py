from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class UploadRequest(BaseModel):
    youtube_url: HttpUrl = Field(..., description='YouTube video URL')
    collection_name: str | None = Field(default=None, description='Optional custom Milvus collection name')


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    video_id: str | None = Field(default=None)
    collection_name: str | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RebuildCollectionRequest(BaseModel):
    youtube_url: HttpUrl
    collection_name: str | None = None


class DeleteCollectionRequest(BaseModel):
    video_id: str | None = Field(default=None)
    collection_name: str | None = Field(default=None)
