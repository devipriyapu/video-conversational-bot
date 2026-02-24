from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

logger = logging.getLogger(__name__)


@dataclass
class DownloadedVideo:
    video_id: str
    title: str
    video_path: Path


class YouTubeService:
    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir

    def extract_video_id(self, youtube_url: str) -> str | None:
        parsed = urlparse(youtube_url.strip())
        host = parsed.netloc.replace('www.', '')

        if host == 'youtu.be':
            video_id = parsed.path.lstrip('/').split('/')[0]
            return video_id or None

        if host in {'youtube.com', 'm.youtube.com'}:
            if parsed.path == '/watch':
                video_id = parse_qs(parsed.query).get('v', [''])[0]
                return video_id or None
            if parsed.path.startswith('/embed/'):
                video_id = parsed.path.split('/')[2] if len(parsed.path.split('/')) > 2 else ''
                return video_id or None

        return None

    def download_video(self, youtube_url: str) -> DownloadedVideo:
        format_candidates = [
            # Prefer progressive streams first (no ffmpeg merge needed).
            'best[ext=mp4][acodec!=none][vcodec!=none]',
            'best[acodec!=none][vcodec!=none]',
            # Fallback to whatever yt-dlp can provide.
            'best',
        ]

        last_error: Exception | None = None
        for fmt in format_candidates:
            ydl_opts = {
                'format': fmt,
                'outtmpl': str(self.upload_dir / '%(id)s.%(ext)s'),
                'quiet': True,
                'noplaylist': True,
                # Improve resilience against YouTube client/signature changes.
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'web', 'tv_embedded'],
                    }
                },
            }
            with YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(youtube_url, download=True)
                except DownloadError as exc:
                    last_error = exc
                    continue

                video_id = info.get('id', '')
                title = info.get('title', video_id)
                ext = info.get('ext', 'mp4')
                candidate_path = self.upload_dir / f'{video_id}.{ext}'

                if not candidate_path.exists():
                    fallback_candidates = [
                        self.upload_dir / f'{video_id}.mp4',
                        self.upload_dir / f'{video_id}.webm',
                        self.upload_dir / f'{video_id}.mkv',
                    ]
                    candidate_path = next((p for p in fallback_candidates if p.exists()), candidate_path)

                if candidate_path.exists():
                    logger.info(
                        'Video downloaded',
                        extra={'video_id': video_id, 'path': str(candidate_path), 'format': fmt},
                    )
                    return DownloadedVideo(video_id=video_id, title=title, video_path=candidate_path)

        if last_error and 'ffmpeg is not installed' in str(last_error).lower():
            raise RuntimeError(
                'ffmpeg is required for the selected YouTube format. '
                'Install ffmpeg (e.g. `brew install ffmpeg`) and retry.'
            ) from last_error

        if last_error and (
            'nsig extraction failed' in str(last_error).lower()
            or 'requested format is not available' in str(last_error).lower()
        ):
            raise RuntimeError(
                'YouTube extractor signature failed for this video. '
                'Upgrade yt-dlp to latest and retry. If it still fails, use browser cookies in yt-dlp.'
            ) from last_error

        raise RuntimeError(
            f'Unable to download video with available formats. Last yt-dlp error: {last_error}'
        ) from last_error
