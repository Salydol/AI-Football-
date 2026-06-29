"""
Event Detection inference — integrates with the Tracking Pipeline.

Takes a sliding window of TrackingFrames and returns detected events.

Usage:
    from fie.models.event_detection.inference import EventDetector
    from fie.models.event_detection.model import FootballTransformerConfig

    detector = EventDetector(checkpoint="checkpoints/best_model.pt")

    # Feed tracking frames one by one
    for tracking_frame in pipeline.process(source):
        result = detector.update(tracking_frame)
        if result:
            print(result)  # {"event": "pass", "confidence": 0.87, ...}
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import math
import torch

from fie.models.event_detection.model import (
    FootballTransformer,
    FootballTransformerConfig,
    load_model,
)
from fie.tracking.pipeline import TrackingFrame


# ---------------------------------------------------------------------------
# Frame → Tensor conversion
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0   # metres (FIFA standard — our tracking uses this)
PITCH_WIDTH = 68.0
MAX_SPEED_KMH = 40.0   # for normalisation


def tracking_frame_to_tensor(
    frames: list[TrackingFrame],
    max_players: int = 23,
    num_features: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of TrackingFrames into model input tensors.

    Returns:
        players: (1, seq_len, max_players, num_features)
        masks:   (1, seq_len, max_players)  True = empty slot
    """
    T = len(frames)
    players_t = torch.zeros(T, max_players, num_features)
    masks_t = torch.ones(T, max_players, dtype=torch.bool)

    for t, frame in enumerate(frames):
        # Slot 0 — ball
        if frame.ball is not None:
            bx = frame.ball.x / PITCH_LENGTH
            by = frame.ball.y / PITCH_WIDTH
            players_t[t, 0, 0] = bx
            players_t[t, 0, 1] = by
            players_t[t, 0, 2] = 0.0   # ball has no speed in our output
            players_t[t, 0, 3] = 0.0
            players_t[t, 0, 4] = 0.0
            players_t[t, 0, 5] = 1.0
            players_t[t, 0, 6] = 1.0   # is_ball flag
            masks_t[t, 0] = False

        # Slots 1..max_players-1 — players
        for i, player in enumerate(frame.players[: max_players - 1], start=1):
            px = player.x / PITCH_LENGTH
            py = player.y / PITCH_WIDTH
            speed = player.speed / MAX_SPEED_KMH
            accel = player.acceleration / 10.0  # rough normalisation
            rad = math.radians(player.direction)
            s, c = math.sin(rad), math.cos(rad)

            players_t[t, i, 0] = px
            players_t[t, i, 1] = py
            players_t[t, i, 2] = speed
            players_t[t, i, 3] = accel
            players_t[t, i, 4] = s
            players_t[t, i, 5] = c
            players_t[t, i, 6] = 0.0  # not ball
            masks_t[t, i] = False

    return players_t.unsqueeze(0), masks_t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Event result
# ---------------------------------------------------------------------------

@dataclass
class EventResult:
    frame_idx: int
    timestamp: float
    event: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "event": self.event,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


# ---------------------------------------------------------------------------
# EventDetector
# ---------------------------------------------------------------------------

class EventDetector:
    """
    Sliding-window event detector that wraps the Football Transformer.

    Maintains a rolling buffer of TrackingFrames and runs inference
    every `stride` frames, returning an EventResult when a non-background
    event exceeds the confidence threshold.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: FootballTransformerConfig | None = None,
        device: str = "cpu",
        seq_len: int = 25,
        stride: int = 5,            # run inference every N frames
        threshold: float = 0.4,     # min confidence to report an event
        suppress_frames: int = 15,  # ignore events within N frames of last one
    ) -> None:
        self.device = device
        self.seq_len = seq_len
        self.stride = stride
        self.threshold = threshold
        self.suppress_frames = suppress_frames

        self._buffer: deque[TrackingFrame] = deque(maxlen=seq_len)
        self._frame_count = 0
        self._last_event_frame = -suppress_frames

        if checkpoint and Path(checkpoint).exists():
            self.model = load_model(str(checkpoint), device=device)
        else:
            # No checkpoint — use random weights (for architecture testing)
            cfg = config or FootballTransformerConfig(seq_len=seq_len)
            self.model = FootballTransformer(cfg)
            self.model.eval()
            if checkpoint:
                import warnings
                warnings.warn(
                    f"Checkpoint not found: {checkpoint}. Using untrained model.",
                    stacklevel=2,
                )

        self.model = self.model.to(device)
        self.event_names = self.model.config.event_names

    def update(self, frame: TrackingFrame) -> EventResult | None:
        """
        Feed one TrackingFrame. Returns an EventResult or None.

        Call this inside your pipeline loop:
            for tracking_frame in pipeline.process(source):
                event = detector.update(tracking_frame)
                if event:
                    handle(event)
        """
        self._buffer.append(frame)
        self._frame_count += 1

        # Wait until buffer is full and it's time to run inference
        if (
            len(self._buffer) < self.seq_len
            or self._frame_count % self.stride != 0
        ):
            return None

        # Suppression: don't fire too quickly
        if self._frame_count - self._last_event_frame < self.suppress_frames:
            return None

        frames = list(self._buffer)
        players, masks = tracking_frame_to_tensor(frames)
        players = players.to(self.device)
        masks = masks.to(self.device)

        result = self.model.predict(players, masks)

        # Ignore background
        if result["event"] == "background":
            return None

        # Threshold check
        if result["confidence"] < self.threshold:
            return None

        self._last_event_frame = self._frame_count

        return EventResult(
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            event=result["event"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )

    def reset(self) -> None:
        """Clear buffer (call between matches)."""
        self._buffer.clear()
        self._frame_count = 0
        self._last_event_frame = -self.suppress_frames
