"""
Video source abstraction.

Supports:
- VideoFileSource  — local mp4 / avi / mkv files
- RTSPSource       — live camera or streaming server

Usage:
    with VideoFileSource("match.mp4") as src:
        for frame_data in src:
            process(frame_data.frame)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from fie.config import settings


@dataclass(slots=True)
class FrameData:
    """Single decoded video frame with metadata."""

    frame: np.ndarray  # BGR, shape (H, W, 3)
    frame_idx: int      # 0-based frame counter
    timestamp: float    # seconds from stream start


class VideoSource(ABC):
    """Abstract video source — iterate to get FrameData objects."""

    @abstractmethod
    def __iter__(self) -> Iterator[FrameData]: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Returns (width, height)."""
        ...

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class VideoFileSource(VideoSource):
    """Read frames from a local video file."""

    def __init__(self, path: str | Path, *, skip_frames: int = 0) -> None:
        """
        Args:
            path: Path to the video file.
            skip_frames: Process every N+1 frames (0 = every frame).
                         E.g. skip_frames=1 processes half the frames.
        """
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self._path}")

        self._skip = skip_frames
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "VideoFileSource: {} | {}x{} @ {:.1f}fps | {} frames",
            self._path.name,
            self._w,
            self._h,
            self._fps,
            self._total,
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._w, self._h)

    @property
    def total_frames(self) -> int:
        return self._total

    def __iter__(self) -> Iterator[FrameData]:
        frame_idx = 0
        read_idx = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            read_idx += 1
            if self._skip and (read_idx - 1) % (self._skip + 1) != 0:
                continue

            frame = _maybe_resize(frame)
            timestamp = frame_idx / self._fps

            yield FrameData(frame=frame, frame_idx=frame_idx, timestamp=timestamp)
            frame_idx += 1

    def close(self) -> None:
        self._cap.release()


class RTSPSource(VideoSource):
    """Read frames from an RTSP stream (camera or streaming server)."""

    def __init__(
        self,
        url: str,
        *,
        reconnect_delay: float = 3.0,
        max_reconnects: int = 5,
    ) -> None:
        """
        Args:
            url: RTSP URL, e.g. rtsp://camera.local:554/stream
            reconnect_delay: Seconds to wait before reconnecting on failure.
            max_reconnects: How many times to retry before giving up.
        """
        self._url = url
        self._reconnect_delay = reconnect_delay
        self._max_reconnects = max_reconnects
        self._cap = self._open()

    def _open(self) -> cv2.VideoCapture:
        logger.info("RTSPSource: connecting to {}", self._url)
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        # Reduce internal buffer so we always get the latest frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot connect to RTSP stream: {self._url}")
        logger.info("RTSPSource: connected | {}x{} @ {:.1f}fps", *self.resolution, self.fps)
        return cap

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def resolution(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def __iter__(self) -> Iterator[FrameData]:
        frame_idx = 0
        reconnects = 0
        start_time = time.monotonic()

        while True:
            ret, frame = self._cap.read()

            if not ret:
                if reconnects >= self._max_reconnects:
                    logger.error("RTSPSource: max reconnects reached, stopping")
                    break

                reconnects += 1
                logger.warning(
                    "RTSPSource: frame grab failed, reconnect {}/{} in {:.0f}s",
                    reconnects,
                    self._max_reconnects,
                    self._reconnect_delay,
                )
                time.sleep(self._reconnect_delay)
                self._cap.release()
                self._cap = self._open()
                continue

            reconnects = 0
            frame = _maybe_resize(frame)
            timestamp = time.monotonic() - start_time

            yield FrameData(frame=frame, frame_idx=frame_idx, timestamp=timestamp)
            frame_idx += 1

    def close(self) -> None:
        self._cap.release()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_resize(frame: np.ndarray) -> np.ndarray:
    """Downscale frame if wider than settings.max_frame_width."""
    max_w = settings.max_frame_width
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / w
    return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def make_source(path_or_url: str, **kwargs: object) -> VideoSource:
    """Factory: auto-detect file vs RTSP from the input string."""
    if path_or_url.startswith("rtsp://") or path_or_url.startswith("rtsps://"):
        return RTSPSource(path_or_url, **kwargs)  # type: ignore[arg-type]
    return VideoFileSource(path_or_url, **kwargs)  # type: ignore[arg-type]
