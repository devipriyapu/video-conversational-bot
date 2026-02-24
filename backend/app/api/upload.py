from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.models.request_models import DeleteCollectionRequest, RebuildCollectionRequest, UploadRequest
from app.models.response_models import GenericResponse, UploadResponse
from app.core.config import get_settings
from app.services.milvus_service import MilvusService
from app.services.pipeline_service import PipelineService
from app.utils.dependencies import get_milvus_service, get_pipeline_service
from yt_dlp.utils import DownloadError

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/upload', tags=['upload'])
settings = get_settings()


def _error_detail(default_message: str, exc: Exception) -> str:
    if settings.app_debug:
        return f'{default_message}: {exc}'
    return default_message


@router.post('', response_model=UploadResponse)
async def upload_video(
    payload: UploadRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
) -> UploadResponse:
    try:
        result = await run_in_threadpool(
            pipeline_service.process_youtube,
            str(payload.youtube_url),
            payload.collection_name,
            False,
        )
        return UploadResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception('Upload processing failed: %s', str(exc))
        raise HTTPException(
            status_code=500,
            detail=_error_detail('Failed to process YouTube URL', exc),
        ) from exc


@router.post('/rebuild', response_model=UploadResponse)
async def rebuild_video_collection(
    payload: RebuildCollectionRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
) -> UploadResponse:
    try:
        result = await run_in_threadpool(
            pipeline_service.process_youtube,
            str(payload.youtube_url),
            payload.collection_name,
            True,
        )
        return UploadResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception('Rebuild processing failed: %s', str(exc))
        raise HTTPException(
            status_code=500,
            detail=_error_detail('Failed to rebuild collection', exc),
        ) from exc


@router.delete('/collection', response_model=GenericResponse)
async def delete_collection(
    payload: DeleteCollectionRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
) -> GenericResponse:
    collection_name = pipeline_service.resolve_collection_name(payload.video_id, payload.collection_name)
    deleted = await run_in_threadpool(milvus_service.drop_collection, collection_name)
    if not deleted:
        raise HTTPException(status_code=404, detail='Collection not found')
    return GenericResponse(message=f'Collection {collection_name} deleted successfully')
