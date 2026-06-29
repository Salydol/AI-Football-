"""
Per-player motion metrics computed from field coordinates.

For each player at each frame we compute:
  - speed         (km/h)
  - acceleration  (m/s²)
  - direction     (degrees, 0=right, 90=up, 180=left, 270=down)

The raw per-frame differences are noisy, so we apply an
exponential moving average (EMA) for speed and a simple
finite-difference for acceleration.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class PlayerMetrics:
    player_id: int
    x: float            # metres
    y: float            # metres
    speed_kmh: float    # km/h
    accel_ms2: float    # m/s²
    direction_deg: float  # 0–360


@dataclass
class _PlayerState:
    """Internal rolling state per player."""
    x: float = 0.0
    y: float = 0.0
    speed_ms: float = 0.0           # smoothed speed in m/s
    prev_speed_ms: float = 0.0      # for acceleration
    last_ts: float | None = None

    # Keep a short history of positions for direction smoothing
    positions: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=5)
    )


# EMA smoothing factor for speed (higher = more responsive, more noise)
_EMA_ALPHA = 0.4


class MetricsCalculator:
    """
    Stateful calculator — call `update` once per frame, per player.

    Maintains history per player_id so it can compute deltas
    between consecutive frames.
    """

    def __init__(self, ema_alpha: float = _EMA_ALPHA) -> None:
        self._alpha = ema_alpha
        self._states: dict[int, _PlayerState] = {}

    def update(
        self,
        player_id: int,
        x: float,
        y: float,
        timestamp: float,
    ) -> PlayerMetrics:
        """
        Feed a new position for `player_id` and return current metrics.

        Args:
            player_id: Tracker-assigned ID.
            x, y: Position in metres on the pitch.
            timestamp: Seconds since match/stream start.
        """
        state = self._states.setdefault(player_id, _PlayerState())

        if state.last_ts is None:
            # First observation — no deltas available yet
            state.x = x
            state.y = y
            state.last_ts = timestamp
            state.positions.append((x, y))
            return PlayerMetrics(
                player_id=player_id,
                x=x,
                y=y,
                speed_kmh=0.0,
                accel_ms2=0.0,
                direction_deg=0.0,
            )

        dt = timestamp - state.last_ts
        if dt <= 0:
            # Duplicate or out-of-order frame — return last known state
            return PlayerMetrics(
                player_id=player_id,
                x=state.x,
                y=state.y,
                speed_kmh=state.speed_ms * 3.6,
                accel_ms2=0.0,
                direction_deg=_direction(state.positions),
            )

        dx = x - state.x
        dy = y - state.y
        dist_m = math.hypot(dx, dy)
        raw_speed_ms = dist_m / dt

        # Clamp unrealistic values (> 40 km/h = tracking glitch)
        raw_speed_ms = min(raw_speed_ms, 11.1)

        # EMA smoothing
        smooth_speed_ms = self._alpha * raw_speed_ms + (1 - self._alpha) * state.speed_ms

        accel = (smooth_speed_ms - state.prev_speed_ms) / dt

        state.x = x
        state.y = y
        state.prev_speed_ms = state.speed_ms
        state.speed_ms = smooth_speed_ms
        state.last_ts = timestamp
        state.positions.append((x, y))

        return PlayerMetrics(
            player_id=player_id,
            x=x,
            y=y,
            speed_kmh=round(smooth_speed_ms * 3.6, 2),
            accel_ms2=round(accel, 3),
            direction_deg=round(_direction(state.positions), 1),
        )

    def reset(self, player_id: int | None = None) -> None:
        """Clear state for one player or all players."""
        if player_id is not None:
            self._states.pop(player_id, None)
        else:
            self._states.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _direction(positions: deque[tuple[float, float]]) -> float:
    """
    Compute movement direction (degrees) from a short position history.

    Uses the vector from oldest to newest point for smoothness.
    Returns 0.0 if there's only one point.
    """
    if len(positions) < 2:
        return 0.0

    x0, y0 = positions[0]
    x1, y1 = positions[-1]
    dx, dy = x1 - x0, y1 - y0

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    # atan2 returns radians, -π to π; convert to 0–360 degrees
    rad = math.atan2(-dy, dx)  # -dy because Y axis points down in image space
    deg = math.degrees(rad) % 360
    return deg
