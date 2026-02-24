from __future__ import annotations

import json
from pathlib import Path

from app.services.audio_service import AudioService
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.rag_service import RagService
from app.services.transcription_service import TranscriptionService
from app.services.youtube_service import YouTubeService


class PipelineService:
    def __init__(
        self,
        youtube_service: YouTubeService,
        audio_service: AudioService,
        transcription_service: TranscriptionService,
        embedding_service: EmbeddingService,
        milvus_service: MilvusService,
        rag_service: RagService,
        transcript_dir: Path,
        create_collection_per_video: bool,
        default_collection: str,
    ) -> None:
        self.youtube_service = youtube_service
        self.audio_service = audio_service
        self.transcription_service = transcription_service
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.rag_service = rag_service
        self.transcript_dir = transcript_dir
        self.create_collection_per_video = create_collection_per_video
        self.default_collection = default_collection

    def resolve_collection_name(self, video_id: str | None, explicit: str | None = None) -> str:
        if explicit:
            return explicit
        if self.create_collection_per_video and video_id:
            return self.milvus_service.collection_name_for_video(video_id)
        return self.default_collection

    def process_youtube(self, youtube_url: str, collection_name: str | None = None, rebuild: bool = False) -> dict:
        downloaded = self.youtube_service.download_video(youtube_url)
        target_collection = self.resolve_collection_name(downloaded.video_id, collection_name)

        if rebuild:
            self.milvus_service.drop_collection(target_collection)

        audio_path = self.audio_service.extract_mp3(downloaded.video_path, downloaded.video_id)
        transcript_text = self.transcription_service.transcribe_audio(audio_path)

        transcript_path = self.transcript_dir / f'{downloaded.video_id}.txt'
        transcript_path.write_text(transcript_text, encoding='utf-8')

        chunks = self.rag_service.chunk_text(transcript_text)
        embeddings = self.embedding_service.embed_batch(chunks)

        metadata = [
            {
                'video_id': downloaded.video_id,
                'title': downloaded.title,
                'chunk_index': idx,
                'audio_path': str(audio_path),
                'transcript_path': str(transcript_path),
            }
            for idx in range(len(chunks))
        ]

        inserted = self.milvus_service.upsert_chunks(target_collection, embeddings, chunks, metadata)

        manifest_path = self.transcript_dir / f'{downloaded.video_id}.json'
        manifest_path.write_text(
            json.dumps(
                {
                    'video_id': downloaded.video_id,
                    'title': downloaded.title,
                    'collection': target_collection,
                    'chunks': inserted,
                },
                indent=2,
            ),
            encoding='utf-8',
        )

        return {
            'video_id': downloaded.video_id,
            'title': downloaded.title,
            'collection_name': target_collection,
            'chunk_count': inserted,
            'transcript_path': str(transcript_path),
        }
