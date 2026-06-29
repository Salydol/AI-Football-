"""
ByteTrack wrapper via the `supervision` library.

Assigns stable IDs to players across frames so that player_id=7
in frame 100 is the same person as player_id=7 in frame 200,
even when they temporarily leave the field of view.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from fie.config import settings
from fie.tracking.detector import Detection, DetectionResult, ObjectClass


class PlayerTracker:
    """
    Runs ByteTrack on player detections.

    Ball and referees are tracked separately (or not tracked)
    because ByteTrack is tuned for multi-object pedestrian tracking.
    """

    def __init__(
        self,
        track_thresh: float | None = None,
        match_thresh: float | None = None,
        frame_rate: int | None = None,
    ) -> None:
        # supervision >= 0.28 renamed ByteTrack → sv.ByteTrack but kept the same API.
        # Suppress the FutureWarning about the old class name.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=track_thresh or settings.bytetrack_track_thresh,
                minimum_matching_threshold=match_thresh or settings.bytetrack_match_thresh,
                frame_rate=frame_rate or settings.bytetrack_frame_rate,
            )

    def update(self, result: DetectionResult) -> DetectionResult:
        """
        Assign tracker_id to every Detection in `result`.

        Modifies `result` in-place and returns it.
        """
        players = result.players + result.referees
        if not players:
            return result

        xyxy = np.array([d.bbox for d in players])
        confidences = np.array([d.confidence for d in players])
        class_ids = np.array([int(d.obj_class) for d in players])

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids,
        )

        tracked = self._tracker.update_with_detections(sv_dets)

        # Map tracked results back to Detection objects.
        # supervision returns a subset of detections that have active tracks.
        if tracked.tracker_id is None:
            return result

        # Build a lookup: bbox → tracker_id (rounded to avoid float drift)
        id_map: dict[tuple[int, ...], int] = {}
        for i in range(len(tracked)):
            key = tuple(int(v) for v in tracked.xyxy[i])
            id_map[key] = int(tracked.tracker_id[i])

        for det in players:
            key = tuple(int(v) for v in det.bbox)
            det.tracker_id = id_map.get(key, -1)

        return result

    def reset(self) -> None:
        """Reset tracker state (call between matches)."""
        self._tracker.reset()


class BallTracker:
    """
    Simple ball tracker — just carries forward the last known position
    when the ball is not detected. No multi-object tracking needed.
    """

    def __init__(self, max_missing_frames: int = 10) -> None:
        self._last: Detection | None = None
        self._missing = 0
        self._max_missing = max_missing_frames
        # Ball always gets tracker_id = 0
        self._BALL_ID = 0

    def update(self, result: DetectionResult) -> DetectionResult:
        if result.ball is not None:
            result.ball.tracker_id = self._BALL_ID
            self._last = result.ball
            self._missing = 0
        else:
            self._missing += 1
            if self._last is not None and self._missing <= self._max_missing:
                # Carry forward last known ball position
                result.ball = Detection(
                    obj_class=ObjectClass.BALL,
                    bbox=self._last.bbox,
                    confidence=self._last.confidence * 0.9,  # decay confidence
                    tracker_id=self._BALL_ID,
                )

        return result
