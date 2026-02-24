from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> None:
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def transcribe_audio(self, audio_path: Path) -> str:
        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        chunk_texts: list[str] = []
        segment_count = 0
        for segment in segments:
            text = segment.text.strip()
            if text:
                chunk_texts.append(text)
            segment_count += 1
            if segment_count % 20 == 0:
                logger.info(
                    'Transcription progress',
                    extra={
                        'audio_path': str(audio_path),
                        'segments_processed': segment_count,
                        'last_time_s': round(float(segment.end), 2),
                    },
                )

        text = ' '.join(chunk_texts)

        if not text:
            raise ValueError('Transcription returned empty text')

        logger.info(
            'Audio transcribed',
            extra={
                'audio_path': str(audio_path),
                'chars': len(text),
                'segments': segment_count,
                'language': getattr(info, 'language', None),
            },
        )
        return text
