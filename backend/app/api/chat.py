from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.models.request_models import ChatRequest
from app.models.response_models import ChatResponse, SourceChunk
from app.services.pipeline_service import PipelineService
from app.services.rag_service import RagService
from app.utils.dependencies import get_pipeline_service, get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/chat', tags=['chat'])
settings = get_settings()


def _error_detail(default_message: str, exc: Exception) -> str:
    if settings.app_debug:
        return f'{default_message}: {exc}'
    return default_message


@router.post('', response_model=ChatResponse)
async def ask_question(
    payload: ChatRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
    rag_service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        collection_name = pipeline_service.resolve_collection_name(payload.video_id, payload.collection_name)
        result = await run_in_threadpool(
            rag_service.answer_question,
            payload.question,
            collection_name,
            payload.top_k,
        )
        return ChatResponse(
            answer=result.answer,
            sources=[SourceChunk(**item) for item in result.sources],
            tokens_used=result.tokens_used,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception('Chat failed: %s', str(exc))
        raise HTTPException(status_code=500, detail=_error_detail('Failed to answer question', exc)) from exc


@router.post('/stream')
async def stream_question(
    payload: ChatRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
    rag_service: RagService = Depends(get_rag_service),
) -> StreamingResponse:
    try:
        collection_name = pipeline_service.resolve_collection_name(payload.video_id, payload.collection_name)
        stream, sources = await run_in_threadpool(
            rag_service.stream_answer,
            payload.question,
            collection_name,
            payload.top_k,
        )

        def event_generator():
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\\n\\n"
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield f"data: {json.dumps({'type': 'token', 'data': delta})}\\n\\n"
            yield "data: {\"type\": \"done\"}\\n\\n"

        return StreamingResponse(event_generator(), media_type='text/event-stream')
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception('Streaming chat failed: %s', str(exc))
        raise HTTPException(status_code=500, detail=_error_detail('Failed to stream answer', exc)) from exc
