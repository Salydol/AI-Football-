"""
Video Clipping — автоматическая нарезка ключевых моментов.

Принцип работы:
  1. EventDetector детектирует событие (пас, удар, перехват...) в кадре N
  2. ClipExtractor открывает видеофайл, прыгает на кадр N - pre_buffer
  3. Считывает pre_buffer + post_buffer кадров и пишет в отдельный MP4
  4. HighlightsBuilder склеивает все клипы в одно highlights-видео

Поддержка:
  - OpenCV VideoWriter (без зависимостей, всегда работает)
  - ffmpeg subprocess (быстрее, лучше качество, если установлен)

Использование:
    extractor = ClipExtractor("match.mp4", output_dir="clips/")
    for event in events:
        path = extractor.extract(event)
        print(f"Saved: {path}")

    builder = HighlightsBuilder(extractor.clips)
    builder.build("highlights.mp4")
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from loguru import logger


# ---------------------------------------------------------------------------
# Event descriptor (минимальный, не зависит от inference.py)
# ---------------------------------------------------------------------------

@dataclass
class ClipEvent:
    """Описание события для нарезки."""
    frame_idx: int
    timestamp: float          # секунды от начала видео
    event_type: str           # "shot", "pass", "tackle", etc.
    confidence: float = 1.0
    player_id: int | None = None
    team: str | None = None


@dataclass
class ClipResult:
    """Результат нарезки одного клипа."""
    event: ClipEvent
    path: str                 # путь к файлу клипа
    duration_s: float
    start_frame: int
    end_frame: int
    width: int
    height: int
    fps: float

    def to_dict(self) -> dict:
        return {
            "event_type": self.event.event_type,
            "timestamp": round(self.event.timestamp, 2),
            "confidence": round(self.event.confidence, 3),
            "player_id": self.event.player_id,
            "team": self.event.team,
            "clip_path": self.path,
            "duration_s": round(self.duration_s, 2),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


# ---------------------------------------------------------------------------
# ClipExtractor
# ---------------------------------------------------------------------------

class ClipExtractor:
    """
    Извлекает видеоклипы вокруг событий из исходного видеофайла.

    Args:
        video_path:     Путь к исходному видео
        output_dir:     Куда сохранять клипы
        pre_seconds:    Секунд до события (default 5)
        post_seconds:   Секунд после события (default 3)
        use_ffmpeg:     Использовать ffmpeg если доступен (лучше качество)
        min_confidence: Минимальная уверенность для нарезки
    """

    def __init__(
        self,
        video_path: str | Path,
        output_dir: str | Path = "clips",
        pre_seconds: float = 5.0,
        post_seconds: float = 3.0,
        use_ffmpeg: bool = True,
        min_confidence: float = 0.0,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.min_confidence = min_confidence
        self.clips: list[ClipResult] = []

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Получаем параметры видео
        cap = cv2.VideoCapture(str(self.video_path))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Проверяем ffmpeg
        self._ffmpeg = use_ffmpeg and shutil.which("ffmpeg") is not None
        if use_ffmpeg and not self._ffmpeg:
            logger.warning("ffmpeg not found — using OpenCV VideoWriter (slower)")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ClipExtractor | video={}x{} @ {}fps | {} frames | ffmpeg={}",
            self._width, self._height, self._fps, self._total_frames, self._ffmpeg,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, event: ClipEvent) -> ClipResult | None:
        """Нарезать один клип вокруг события."""
        if event.confidence < self.min_confidence:
            return None

        pre_frames = int(self.pre_seconds * self._fps)
        post_frames = int(self.post_seconds * self._fps)

        start = max(0, event.frame_idx - pre_frames)
        end = min(self._total_frames - 1, event.frame_idx + post_frames)

        safe_type = event.event_type.replace(" ", "_")
        filename = f"{safe_type}_t{event.timestamp:.1f}s_conf{event.confidence:.2f}.mp4"
        out_path = self.output_dir / filename

        if self._ffmpeg:
            success = self._extract_ffmpeg(start, end, out_path)
        else:
            success = self._extract_opencv(start, end, out_path)

        if not success:
            logger.warning("Failed to extract clip for event at frame {}", event.frame_idx)
            return None

        duration = (end - start) / self._fps
        result = ClipResult(
            event=event,
            path=str(out_path),
            duration_s=duration,
            start_frame=start,
            end_frame=end,
            width=self._width,
            height=self._height,
            fps=self._fps,
        )
        self.clips.append(result)
        logger.info(
            "Clip saved: {} | {:.1f}s | frames {}-{}",
            out_path.name, duration, start, end,
        )
        return result

    def extract_batch(self, events: list[ClipEvent]) -> list[ClipResult]:
        """Нарезать клипы для списка событий."""
        results = []
        for event in events:
            r = self.extract(event)
            if r:
                results.append(r)
        return results

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ------------------------------------------------------------------
    # Internal: ffmpeg backend (быстрее, лучше качество)
    # ------------------------------------------------------------------

    def _extract_ffmpeg(self, start_frame: int, end_frame: int, out_path: Path) -> bool:
        start_t = start_frame / self._fps
        duration = (end_frame - start_frame) / self._fps
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_t:.3f}",
            "-i", str(self.video_path),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-an",                  # без звука
            "-loglevel", "error",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            return out_path.exists() and out_path.stat().st_size > 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("ffmpeg error: {}", e)
            return False

    # ------------------------------------------------------------------
    # Internal: OpenCV backend (нет зависимостей)
    # ------------------------------------------------------------------

    def _extract_opencv(self, start_frame: int, end_frame: int, out_path: Path) -> bool:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path), fourcc, self._fps, (self._width, self._height)
        )

        frame_idx = start_frame
        ok = True
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        ok = out_path.exists() and out_path.stat().st_size > 0
        return ok


# ---------------------------------------------------------------------------
# HighlightsBuilder
# ---------------------------------------------------------------------------

class HighlightsBuilder:
    """
    Собирает highlights-видео из списка клипов.

    Args:
        clips:          Список ClipResult
        sort_by:        'time' (хронологически) | 'confidence' (по уверенности)
        add_title:      Показывать текстовый оверлей с типом события
        title_color:    BGR цвет текста (default: жёлтый)
    """

    def __init__(
        self,
        clips: list[ClipResult],
        sort_by: str = "time",
        add_title: bool = True,
        title_color: tuple[int, int, int] = (0, 255, 255),
    ) -> None:
        self.clips = sorted(
            clips,
            key=lambda c: c.event.timestamp if sort_by == "time" else -c.event.confidence,
        )
        self.add_title = add_title
        self.title_color = title_color

    def build(self, output_path: str | Path, use_ffmpeg: bool = True) -> Path:
        """
        Собрать highlights из всех клипов.

        Args:
            output_path:    Путь к итоговому видео
            use_ffmpeg:     Использовать ffmpeg concat если доступен

        Returns:
            Path к итоговому видео
        """
        output_path = Path(output_path)
        if not self.clips:
            raise ValueError("No clips to build highlights from")

        if use_ffmpeg and shutil.which("ffmpeg"):
            return self._build_ffmpeg(output_path)
        else:
            return self._build_opencv(output_path)

    def total_duration(self) -> float:
        return sum(c.duration_s for c in self.clips)

    # ------------------------------------------------------------------

    def _build_ffmpeg(self, output_path: Path) -> Path:
        """ffmpeg concat — без перекодирования если форматы совпадают."""
        # Создаём список файлов для ffmpeg concat
        list_path = output_path.parent / "_concat_list.txt"
        with open(list_path, "w") as f:
            for clip in self.clips:
                f.write(f"file '{Path(clip.path).resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-loglevel", "error",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg concat failed: {}", e.stderr.decode())
            raise
        finally:
            list_path.unlink(missing_ok=True)

        logger.info(
            "Highlights built: {} | {} clips | {:.1f}s total",
            output_path.name, len(self.clips), self.total_duration(),
        )
        return output_path

    def _build_opencv(self, output_path: Path) -> Path:
        """OpenCV frame-by-frame склейка с оверлеем."""
        if not self.clips:
            raise ValueError("No clips")

        first = self.clips[0]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, first.fps, (first.width, first.height)
        )

        for clip in self.clips:
            cap = cv2.VideoCapture(clip.path)
            frame_n = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.add_title and frame_n < int(clip.fps * 2):
                    # Показываем тип события первые 2 секунды клипа
                    label = f"{clip.event.event_type.upper()}  {clip.event.confidence:.0%}"
                    cv2.putText(
                        frame, label,
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, (0, 0, 0), 4,        # тень
                    )
                    cv2.putText(
                        frame, label,
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, self.title_color, 2,  # текст
                    )
                writer.write(frame)
                frame_n += 1
            cap.release()

        writer.release()
        logger.info(
            "Highlights built: {} | {} clips | {:.1f}s total",
            output_path.name, len(self.clips), self.total_duration(),
        )
        return output_path


# ---------------------------------------------------------------------------
# Helper: конвертация EventResult → ClipEvent
# ---------------------------------------------------------------------------

def event_result_to_clip_event(result: object) -> ClipEvent:
    """
    Конвертирует fie.models.event_detection.inference.EventResult в ClipEvent.
    Работает duck-typing — не требует импорта inference.
    """
    return ClipEvent(
        frame_idx=getattr(result, "frame_idx", 0),
        timestamp=getattr(result, "timestamp", 0.0),
        event_type=getattr(result, "event", "unknown"),
        confidence=getattr(result, "confidence", 1.0),
        player_id=getattr(result, "player_id", None),
    )
