"""
Video Index — индексация событий видеозаписи.

Принцип работы:
  1. VideoIndexer запускает EventDetector на видео (или принимает готовые события)
  2. Каждое событие сохраняется с таймстемпом, типом, уверенностью, описанием
  3. Индекс хранится в data/search_index/{video_id}.json
  4. QueryEngine ищет по индексу (rule-based или LLM)

Структура индекса:
  {
    "video_id": "match_lokomotiv_cska",
    "video_path": "match.mp4",
    "indexed_at": "2024-...",
    "total_frames": 159488,
    "fps": 25.0,
    "duration_s": 6379.5,
    "events": [
      {
        "idx": 0,
        "frame_idx": 3456,
        "timestamp_s": 138.2,
        "minute": 2,
        "second": 18,
        "half": 1,
        "event_type": "shot",
        "confidence": 0.81,
        "player_id": 9,
        "description": "Shot by #9 at 02:18 (1st half)"
      }
    ]
  }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import cv2
from loguru import logger


# ---------------------------------------------------------------------------
# Indexed Event
# ---------------------------------------------------------------------------

@dataclass
class IndexedEvent:
    """Одно проиндексированное событие."""
    idx: int                  # порядковый номер в индексе
    frame_idx: int
    timestamp_s: float
    minute: int
    second: int
    half: int                 # 1 или 2
    event_type: str           # shot / pass / tackle / dribble / etc.
    confidence: float
    player_id: int | None
    description: str          # человекочитаемое описание

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IndexedEvent":
        return cls(**d)

    def matches_time_filter(
        self,
        half: int | None = None,
        minute_from: int | None = None,
        minute_to: int | None = None,
    ) -> bool:
        if half is not None and self.half != half:
            return False
        if minute_from is not None and self.minute < minute_from:
            return False
        if minute_to is not None and self.minute > minute_to:
            return False
        return True


# ---------------------------------------------------------------------------
# VideoIndex
# ---------------------------------------------------------------------------

class VideoIndex:
    """Индекс событий одного видеофайла."""

    def __init__(
        self,
        video_id: str,
        video_path: str = "",
        fps: float = 25.0,
        total_frames: int = 0,
        match_duration_min: float = 90.0,
    ) -> None:
        self.video_id = video_id
        self.video_path = video_path
        self.fps = fps
        self.total_frames = total_frames
        self.match_duration_min = match_duration_min
        self.indexed_at = datetime.now().isoformat()
        self.events: list[IndexedEvent] = []

    @property
    def duration_s(self) -> float:
        return self.total_frames / max(self.fps, 1)

    def add_event(
        self,
        frame_idx: int,
        event_type: str,
        confidence: float,
        player_id: int | None = None,
        video_offset_s: float = 0.0,
    ) -> IndexedEvent:
        """Добавить событие в индекс."""
        ts = frame_idx / max(self.fps, 1) + video_offset_s
        # Определяем минуту матча
        match_minute = int(ts / 60)
        match_second = int(ts % 60)
        half = 2 if match_minute >= 45 else 1

        pid_str = f" by #{player_id}" if player_id else ""
        half_str = f"{half}st" if half == 1 else "2nd"
        description = (
            f"{event_type.upper()}{pid_str} "
            f"at {match_minute:02d}:{match_second:02d} ({half_str} half)"
        )

        event = IndexedEvent(
            idx=len(self.events),
            frame_idx=frame_idx,
            timestamp_s=round(ts, 2),
            minute=match_minute,
            second=match_second,
            half=half,
            event_type=event_type,
            confidence=round(confidence, 3),
            player_id=player_id,
            description=description,
        )
        self.events.append(event)
        return event

    def search(
        self,
        event_types: list[str] | None = None,
        half: int | None = None,
        minute_from: int | None = None,
        minute_to: int | None = None,
        player_id: int | None = None,
        min_confidence: float = 0.0,
    ) -> list[IndexedEvent]:
        """Структурированный поиск по индексу."""
        results = []
        for e in self.events:
            if event_types and e.event_type not in event_types:
                continue
            if not e.matches_time_filter(half, minute_from, minute_to):
                continue
            if player_id is not None and e.player_id != player_id:
                continue
            if e.confidence < min_confidence:
                continue
            results.append(e)
        return results

    def save(self, save_dir: str | Path = "data/search_index") -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{self.video_id}.json"
        data = {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "match_duration_min": self.match_duration_min,
            "indexed_at": self.indexed_at,
            "duration_s": round(self.duration_s, 1),
            "total_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Video index saved: {} | {} events", path.name, len(self.events))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VideoIndex":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        idx = cls(
            video_id=data["video_id"],
            video_path=data.get("video_path", ""),
            fps=data.get("fps", 25.0),
            total_frames=data.get("total_frames", 0),
            match_duration_min=data.get("match_duration_min", 90.0),
        )
        idx.indexed_at = data.get("indexed_at", "")
        idx.events = [IndexedEvent.from_dict(e) for e in data.get("events", [])]
        return idx

    @classmethod
    def load_or_create(
        cls,
        video_id: str,
        save_dir: str | Path = "data/search_index",
        **kwargs,
    ) -> "VideoIndex":
        path = Path(save_dir) / f"{video_id}.json"
        if path.exists():
            return cls.load(path)
        return cls(video_id=video_id, **kwargs)

    def stats(self) -> dict:
        from collections import Counter
        type_counts = Counter(e.event_type for e in self.events)
        return {
            "video_id": self.video_id,
            "total_events": len(self.events),
            "duration_s": round(self.duration_s, 1),
            "indexed_at": self.indexed_at,
            "event_types": dict(type_counts),
        }


# ---------------------------------------------------------------------------
# VideoIndexer — строит индекс из EventDetector
# ---------------------------------------------------------------------------

class VideoIndexer:
    """
    Запускает EventDetector на видео и строит VideoIndex.

    Args:
        video_path:         Путь к видео
        video_id:           Уникальный ID (default: имя файла без расширения)
        device:             "cuda" / "cpu"
        checkpoint_path:    Путь к чекпоинту EventDetector
        save_dir:           Куда сохранять индекс
        match_duration_min: Длительность матча для расчёта таймов (default 90)
        video_offset_s:     Смещение начала матча в видео (если есть интро)
        min_confidence:     Минимальная уверенность события для индексации
    """

    def __init__(
        self,
        video_path: str | Path,
        video_id: str | None = None,
        device: str = "cuda",
        checkpoint_path: str = "checkpoints/best_model-v2.ckpt",
        save_dir: str | Path = "data/search_index",
        match_duration_min: float = 90.0,
        video_offset_s: float = 0.0,
        min_confidence: float = 0.35,
    ) -> None:
        self.video_path = Path(video_path)
        self.video_id = video_id or self.video_path.stem
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.save_dir = Path(save_dir)
        self.match_duration_min = match_duration_min
        self.video_offset_s = video_offset_s
        self.min_confidence = min_confidence

    def build(
        self,
        max_frames: int = 0,
        start_frame: int = 0,
    ) -> VideoIndex:
        """
        Проиндексировать видео.

        Args:
            max_frames:     Максимум кадров (0 = всё видео)
            start_frame:    Начать с этого кадра

        Returns:
            VideoIndex с найденными событиями
        """
        from fie.tracking.detector import FootballDetector
        from fie.tracking.pipeline import TrackingPipeline
        from fie.tracking.calibration import IdentityCalibration
        from fie.tracking.source import make_source
        from fie.models.event_detection.inference import EventDetector

        logger.info("Indexing video: {} | start={} | max={}", self.video_path.name, start_frame, max_frames or "all")

        cap_tmp = cv2.VideoCapture(str(self.video_path))
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_tmp.release()

        index = VideoIndex(
            video_id=self.video_id,
            video_path=str(self.video_path),
            fps=fps,
            total_frames=total_frames,
            match_duration_min=self.match_duration_min,
        )

        detector = FootballDetector(device=self.device)
        pipeline = TrackingPipeline(
            calibration=IdentityCalibration(),
            detector=detector,
        )
        event_det = EventDetector(
            self.checkpoint_path,
            device=self.device,
            threshold=self.min_confidence,
        )

        source = make_source(str(self.video_path))
        if start_frame > 0:
            source._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        t0 = time.monotonic()

        with source:
            for raw in source:
                if max_frames and frame_count >= max_frames:
                    break

                tracking_frame = pipeline.process_frame(raw)
                result = event_det.update(tracking_frame)

                if result and result.event != "background":
                    index.add_event(
                        frame_idx=raw.frame_idx + start_frame,
                        event_type=result.event,
                        confidence=result.confidence,
                        player_id=getattr(result, "player_id", None),
                        video_offset_s=self.video_offset_s,
                    )

                frame_count += 1
                if frame_count % 500 == 0:
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "Indexing: frame={} | events={} | {:.1f}fps",
                        frame_count, len(index.events), frame_count / elapsed,
                    )

        logger.info(
            "Indexing complete: {} events in {} frames ({:.1f}s)",
            len(index.events), frame_count, time.monotonic() - t0,
        )
        return index

    def build_from_events(self, events: list[dict]) -> VideoIndex:
        """
        Построить индекс из готового списка событий (без запуска детектора).

        Полезно если детекция уже была выполнена отдельно.

        Args:
            events: list[{frame_idx, event_type, confidence, player_id?}]
        """
        cap_tmp = cv2.VideoCapture(str(self.video_path))
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_tmp.release()

        index = VideoIndex(
            video_id=self.video_id,
            video_path=str(self.video_path),
            fps=fps,
            total_frames=total_frames,
            match_duration_min=self.match_duration_min,
        )

        for e in events:
            if e.get("confidence", 1.0) >= self.min_confidence:
                index.add_event(
                    frame_idx=e["frame_idx"],
                    event_type=e.get("event_type", "unknown"),
                    confidence=e.get("confidence", 1.0),
                    player_id=e.get("player_id"),
                    video_offset_s=self.video_offset_s,
                )

        return index
