from __future__ import annotations

from functools import lru_cache
import logging

from openai import OpenAI

from app.core.config import Settings, get_settings
from app.services.audio_service import AudioService
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.pipeline_service import PipelineService
from app.services.rag_service import RagService
from app.services.transcription_service import TranscriptionService
from app.services.youtube_service import YouTubeService

logger = logging.getLogger(__name__)


def _resolved_api_key() -> str:
    settings = get_settings()
    key = settings.openai_api_key.strip()
    return key if key else 'local-dev-key'


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=_resolved_api_key(), base_url=settings.openai_base_url)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    model_name = settings.embedding_model
    if model_name.startswith('text-embedding-'):
        logger.warning(
            'Invalid sentence-transformers model configured (%s). Falling back to all-MiniLM-L6-v2.',
            model_name,
        )
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    return EmbeddingService(model_name, settings.embedding_device)


@lru_cache(maxsize=1)
def get_milvus_service() -> MilvusService:
    settings = get_settings()
    return MilvusService(
        uri=settings.milvus_uri,
        default_collection=settings.milvus_default_collection,
        dimension=settings.milvus_dimension,
        top_k=settings.milvus_top_k,
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(
        embedding_service=get_embedding_service(),
        milvus_service=get_milvus_service(),
        openai_client=get_openai_client(),
        chat_model=settings.chat_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_context_chunks=settings.max_context_chunks,
    )


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    settings = get_settings()
    return PipelineService(
        youtube_service=YouTubeService(settings.upload_dir),
        audio_service=AudioService(settings.audio_dir),
        transcription_service=TranscriptionService(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.whisper_beam_size,
            vad_filter=settings.whisper_vad_filter,
        ),
        embedding_service=get_embedding_service(),
        milvus_service=get_milvus_service(),
        rag_service=get_rag_service(),
        transcript_dir=settings.transcript_dir,
        create_collection_per_video=settings.milvus_create_collection_per_video,
        default_collection=settings.milvus_default_collection,
    )


def get_app_settings() -> Settings:
    return get_settings()
