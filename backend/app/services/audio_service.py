from __future__ import annotations

from pathlib import Path

from moviepy.editor import VideoFileClip


class AudioService:
    def __init__(self, audio_dir: Path) -> None:
        self.audio_dir = audio_dir

    def extract_mp3(self, video_path: Path, video_id: str) -> Path:
        output = self.audio_dir / f'{video_id}.mp3'
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                raise ValueError('No audio stream found in video')
            clip.audio.write_audiofile(str(output), codec='mp3', logger=None)
        return output
