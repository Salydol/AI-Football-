"""
Main Tracking Pipeline.

Orchestrates:
  VideoSource → Detector → Tracker → Calibration → MetricsCalculator

Output per frame:
  TrackingFrame — list of TrackedPlayer + optional ball position

Usage:
    from fie.tracking.pipeline import TrackingPipeline
    from fie.tracking.source import VideoFileSource
    from fie.tracking.calibration import ManualCalibration

    cal = ManualCalibration(pixel_points=[...], field_points=[...])
    pipeline = TrackingPipeline(calibration=cal)

    with VideoFileSource("match.mp4") as src:
        for tracking_frame in pipeline.process(src):
            print(tracking_frame)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from loguru import logger

from fie.tracking.calibration import FieldCalibration, IdentityCalibration
from fie.tracking.detector import FootballDetector
from fie.tracking.metrics import MetricsCalculator, PlayerMetrics
from fie.tracking.source import FrameData, VideoSource
from fie.tracking.tracker import BallTracker, PlayerTracker


@dataclass(slots=True)
class TrackedPlayer:
    """Single player's state for one frame — this is the final output unit."""
    player_id: int
    x: float            # metres on pitch
    y: float            # metres on pitch
    speed: float        # km/h
    acceleration: float # m/s²
    direction: float    # degrees


@dataclass(slots=True)
class TrackedBall:
    x: float   # metres on pitch
    y: float   # metres on pitch
    confidence: float


@dataclass
class TrackingFrame:
    """Complete tracking output for one video frame."""
    frame_idx: int
    timestamp: float    # seconds
    players: list[TrackedPlayer] = field(default_factory=list)
    ball: TrackedBall | None = None

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "players": [
                {
                    "player_id": p.player_id,
                    "x": round(p.x, 2),
                    "y": round(p.y, 2),
                    "speed": round(p.speed, 2),
                    "acceleration": round(p.acceleration, 3),
                    "direction": round(p.direction, 1),
                }
                for p in self.players
            ],
            "ball": (
                {
                    "x": round(self.ball.x, 2),
                    "y": round(self.ball.y, 2),
                    "confidence": round(self.ball.confidence, 3),
                }
                if self.ball
                else None
            ),
        }


class TrackingPipeline:
    """
    End-to-end tracking pipeline.

    Args:
        calibration: Field calibration instance. Defaults to IdentityCalibration
                     (pixel coords) if not provided — useful for quick tests.
        detector: FootballDetector instance. Created with defaults if not provided.
        log_interval: Log FPS stats every N frames (0 = never).
    """

    def __init__(
        self,
        calibration: FieldCalibration | None = None,
        detector: FootballDetector | None = None,
        log_interval: int = 500,
    ) -> None:
        self._calibration = calibration or IdentityCalibration()
        self._detector = detector or FootballDetector()
        self._player_tracker = PlayerTracker()
        self._ball_tracker = BallTracker()
        self._metrics = MetricsCalculator()
        self._log_interval = log_interval

    def process(self, source: VideoSource) -> Iterator[TrackingFrame]:
        """
        Iterate over a VideoSource and yield TrackingFrame objects.

        This is a generator — frames are processed lazily so you can
        pipe the output to downstream models without loading everything
        into memory.
        """
        self._reset()
        t_start = time.monotonic()

        for frame_data in source:
            tracking_frame = self._process_frame(frame_data)
            yield tracking_frame

            if self._log_interval and frame_data.frame_idx % self._log_interval == 0:
                elapsed = time.monotonic() - t_start or 1e-9
                fps = (frame_data.frame_idx + 1) / elapsed
                logger.info(
                    "Frame {:>6} | t={:.1f}s | {:.1f} fps | {} players",
                    frame_data.frame_idx,
                    frame_data.timestamp,
                    fps,
                    len(tracking_frame.players),
                )

    def process_frame(self, frame_data: FrameData) -> TrackingFrame:
        """Process a single frame (useful for API handlers)."""
        return self._process_frame(frame_data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_frame(self, frame_data: FrameData) -> TrackingFrame:
        # 1. Detect
        det_result = self._detector.detect(
            frame_data.frame,
            frame_data.frame_idx,
            frame_data.timestamp,
        )

        # 2. Track (assign stable IDs)
        self._player_tracker.update(det_result)
        self._ball_tracker.update(det_result)

        # 3. Convert to field coords + compute metrics
        tracked_players: list[TrackedPlayer] = []

        for det in det_result.players + det_result.referees:
            if det.tracker_id < 0:
                continue  # not confirmed by tracker yet

            # Use foot point (bottom-centre) for ground-plane projection
            px, py = det.foot_point
            field_pt = self._calibration.to_field(px, py)
            if field_pt is None:
                continue  # outside pitch boundary

            metrics: PlayerMetrics = self._metrics.update(
                player_id=det.tracker_id,
                x=field_pt.x,
                y=field_pt.y,
                timestamp=frame_data.timestamp,
            )

            tracked_players.append(
                TrackedPlayer(
                    player_id=det.tracker_id,
                    x=metrics.x,
                    y=metrics.y,
                    speed=metrics.speed_kmh,
                    acceleration=metrics.accel_ms2,
                    direction=metrics.direction_deg,
                )
            )

        # 4. Ball
        tracked_ball: TrackedBall | None = None
        if det_result.ball is not None:
            px, py = det_result.ball.center
            field_pt = self._calibration.to_field(px, py)
            if field_pt is not None:
                tracked_ball = TrackedBall(
                    x=field_pt.x,
                    y=field_pt.y,
                    confidence=det_result.ball.confidence,
                )

        return TrackingFrame(
            frame_idx=frame_data.frame_idx,
            timestamp=frame_data.timestamp,
            players=tracked_players,
            ball=tracked_ball,
        )

    def _reset(self) -> None:
        self._player_tracker.reset()
        self._metrics.reset()
