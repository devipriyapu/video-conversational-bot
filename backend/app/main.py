from __future__ import annotations

import os

# Set before importing modules that may indirectly load torch/OpenMP runtimes.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.response_models import HealthResponse

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(title=settings.app_name, debug=settings.app_debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health', response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status='ok', app=settings.app_name)


app.include_router(upload_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
