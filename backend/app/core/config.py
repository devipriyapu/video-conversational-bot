from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = Field(default='video-conversational-bot', alias='APP_NAME')
    app_env: str = Field(default='development', alias='APP_ENV')
    app_debug: bool = Field(default=False, alias='APP_DEBUG')
    api_prefix: str = Field(default='/api/v1', alias='API_PREFIX')
    cors_origins: str = Field(default='http://localhost:4200', alias='CORS_ORIGINS')

    openai_api_key: str = Field(default='', alias='OPENAI_API_KEY')
    openai_base_url: str = Field(default='https://api.openai.com/v1', alias='OPENAI_BASE_URL')
    chat_model: str = Field(default='gpt-4o-mini', alias='CHAT_MODEL')

    embedding_model: str = Field(default='sentence-transformers/all-MiniLM-L6-v2', alias='EMBEDDING_MODEL')
    embedding_device: str = Field(default='cpu', alias='EMBEDDING_DEVICE')

    milvus_uri: str = Field(default='./milvus.db', alias='APP_MILVUS_URI')
    milvus_default_collection: str = Field(default='video_chunks', alias='MILVUS_DEFAULT_COLLECTION')
    milvus_dimension: int = Field(default=384, alias='MILVUS_DIMENSION')
    milvus_top_k: int = Field(default=5, alias='MILVUS_TOP_K')
    milvus_create_collection_per_video: bool = Field(default=True, alias='MILVUS_CREATE_COLLECTION_PER_VIDEO')

    upload_dir: Path = Field(default=Path('./data/uploads'), alias='UPLOAD_DIR')
    audio_dir: Path = Field(default=Path('./data/audio'), alias='AUDIO_DIR')
    transcript_dir: Path = Field(default=Path('./data/transcripts'), alias='TRANSCRIPT_DIR')

    whisper_model: str = Field(default='small', alias='WHISPER_MODEL')
    whisper_compute_type: str = Field(default='int8', alias='WHISPER_COMPUTE_TYPE')
    whisper_device: str = Field(default='cpu', alias='WHISPER_DEVICE')
    whisper_beam_size: int = Field(default=1, alias='WHISPER_BEAM_SIZE')
    whisper_vad_filter: bool = Field(default=True, alias='WHISPER_VAD_FILTER')

    chunk_size: int = Field(default=1200, alias='CHUNK_SIZE')
    chunk_overlap: int = Field(default=200, alias='CHUNK_OVERLAP')
    max_context_chunks: int = Field(default=6, alias='MAX_CONTEXT_CHUNKS')

    log_level: str = Field(default='INFO', alias='LOG_LEVEL')

    @property
    def cors_origin_list(self) -> List[str]:
        return [item.strip() for item in self.cors_origins.split(',') if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.transcript_dir.mkdir(parents=True, exist_ok=True)
    return settings
