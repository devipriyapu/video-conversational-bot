from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# pymilvus reads MILVUS_URI during module import and expects an HTTP URI.
# It also calls load_dotenv() internally, so we must pre-set MILVUS_URI to a valid URL
# to prevent import-time crashes when .env contains a file path (Milvus Lite mode).
legacy_uri = os.getenv('MILVUS_URI', '')
if legacy_uri and not legacy_uri.startswith(('http://', 'https://')):
    os.environ.setdefault('APP_MILVUS_URI', legacy_uri)
os.environ.setdefault('MILVUS_URI', 'http://localhost:19530')

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

logger = logging.getLogger(__name__)


class MilvusService:
    def __init__(self, uri: str, default_collection: str, dimension: int, top_k: int) -> None:
        self.uri = uri
        self.default_collection = self._sanitize_collection_name(default_collection)
        self.dimension = dimension
        self.top_k = top_k

        # Milvus Lite creates local Unix sockets under TMPDIR; ensure a writable path.
        tmp_dir = Path('./data/tmp').resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('TMPDIR', str(tmp_dir))

        connections.connect(uri=self.uri)

    @staticmethod
    def _sanitize_collection_name(name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if not sanitized:
            sanitized = 'video_chunks'
        return sanitized[:255]

    def collection_name_for_video(self, video_id: str) -> str:
        return self._sanitize_collection_name(f'video_{video_id}')

    def _build_schema(self, dimension: int) -> CollectionSchema:
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='metadata', dtype=DataType.JSON),
        ]
        return CollectionSchema(fields=fields, description='Video transcript chunks')

    def ensure_collection(self, collection_name: str, dimension: int) -> Collection:
        collection_name = self._sanitize_collection_name(collection_name)

        def _create_supported_index(collection: Collection) -> None:
            index_candidates = [
                {
                    'index_type': 'HNSW',
                    'metric_type': 'COSINE',
                    'params': {'M': 8, 'efConstruction': 64},
                },
                {
                    'index_type': 'AUTOINDEX',
                    'metric_type': 'COSINE',
                    'params': {},
                },
                {
                    'index_type': 'IVF_FLAT',
                    'metric_type': 'COSINE',
                    'params': {'nlist': 1024},
                },
                {
                    'index_type': 'FLAT',
                    'metric_type': 'COSINE',
                    'params': {},
                },
            ]
            last_error: Exception | None = None
            for params in index_candidates:
                try:
                    collection.create_index(field_name='embedding', index_params=params)
                    logger.info(
                        'Created Milvus index',
                        extra={'collection': collection.name, 'index_type': params['index_type']},
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    logger.warning(
                        'Index type not supported, trying next',
                        extra={'collection': collection.name, 'index_type': params['index_type'], 'error': str(exc)},
                    )
            raise RuntimeError(f'Unable to create a supported index for collection {collection.name}: {last_error}')

        if not utility.has_collection(collection_name):
            schema = self._build_schema(dimension)
            collection = Collection(name=collection_name, schema=schema)
            _create_supported_index(collection)
            logger.info('Created Milvus collection', extra={'collection': collection_name, 'dim': dimension})
        else:
            collection = Collection(name=collection_name)
            try:
                existing_indexes = collection.indexes
            except Exception:  # noqa: BLE001
                existing_indexes = []
            if not existing_indexes:
                _create_supported_index(collection)

        collection.load()
        return collection

    def upsert_chunks(
        self,
        collection_name: str,
        embeddings: list[list[float]],
        chunks: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not embeddings:
            return 0

        dimension = len(embeddings[0])
        collection = self.ensure_collection(collection_name, dimension)

        payload = [embeddings, chunks, metadatas]
        result = collection.insert(payload)
        collection.flush()
        return len(result.primary_keys)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        collection_name = self._sanitize_collection_name(collection_name)
        if not utility.has_collection(collection_name):
            return []

        collection = Collection(name=collection_name)
        collection.load()

        search_result = collection.search(
            data=[query_vector],
            anns_field='embedding',
            param={'metric_type': 'COSINE', 'params': {'ef': 64}},
            limit=top_k or self.top_k,
            output_fields=['text', 'metadata'],
        )

        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for hits in search_result:
            for hit in hits:
                metadata = hit.entity.get('metadata')
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {'raw': metadata}
                text = hit.entity.get('text')
                safe_metadata = metadata or {}
                dedupe_key = (
                    str(safe_metadata.get('video_id', '')),
                    str(safe_metadata.get('chunk_index', '')),
                    str(text or ''),
                )
                if dedupe_key in seen:
                    continue

                seen.add(dedupe_key)
                rows.append(
                    {
                        'id': hit.id,
                        'score': float(hit.score),
                        'text': text,
                        'metadata': safe_metadata,
                    }
                )
        return rows

    def drop_collection(self, collection_name: str) -> bool:
        collection_name = self._sanitize_collection_name(collection_name)
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            return True
        return False

    def collection_size(self, collection_name: str) -> int:
        collection_name = self._sanitize_collection_name(collection_name)
        if not utility.has_collection(collection_name):
            return 0

        collection = Collection(name=collection_name)
        return int(collection.num_entities)
