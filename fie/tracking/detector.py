"""
YOLO-based object detector for football scenes.

Detects:
  - players (both teams)
  - goalkeeper
  - referee
  - ball

Uses ultralytics YOLOv8. The standard COCO model already detects
"person" (class 0) and "sports ball" (class 32), so no fine-tuning
is required to get started. Fine-tuned football weights can be
swapped in via YOLO_MODEL env var.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import supervision as sv
from loguru import logger
from ultralytics import YOLO

from fie.config import settings


class ObjectClass(IntEnum):
    """Detected object types."""
    PLAYER = 0
    BALL = 1
    REFEREE = 2
    GOALKEEPER = 3  # distinguished post-hoc via team color clustering


@dataclass(slots=True)
class Detection:
    """Single detected object on one frame."""

    obj_class: ObjectClass
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    confidence: float
    # Filled in by the tracker after this step:
    tracker_id: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of bounding box — better ground-plane proxy than center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass
class DetectionResult:
    """All detections for a single frame."""

    frame_idx: int
    timestamp: float
    players: list[Detection] = field(default_factory=list)
    ball: Detection | None = None
    referees: list[Detection] = field(default_factory=list)

    @property
    def all_objects(self) -> list[Detection]:
        objs = list(self.players) + list(self.referees)
        if self.ball:
            objs.append(self.ball)
        return objs


# COCO class IDs used by the base YOLOv8 model
_COCO_PERSON = 0
_COCO_SPORTS_BALL = 32


class FootballDetector:
    """
    Wraps YOLOv8 and filters / maps detections to football objects.

    If you later switch to a fine-tuned football model (e.g. from
    Roboflow Universe) that has custom class IDs, subclass this and
    override `_map_detections`.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        device: str | None = None,
    ) -> None:
        model_path = model_path or settings.yolo_model
        self._conf = confidence or settings.yolo_confidence
        self._device = device or settings.yolo_device

        logger.info("Loading YOLO model: {} on {}", model_path, self._device)
        self._model = YOLO(model_path)
        self._model.to(self._device)
        logger.info("YOLO model ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> DetectionResult:
        """Run inference on a single BGR frame."""
        # Use a lower confidence threshold so the ball (small object) is not missed.
        # We filter by class after inference.
        raw = self._model(
            frame,
            conf=min(self._conf, 0.15),  # lower threshold catches small balls
            verbose=False,
            device=self._device,
        )[0]

        sv_detections = sv.Detections.from_ultralytics(raw)
        return self._map_detections(sv_detections, frame_idx, timestamp)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _map_detections(
        self,
        dets: sv.Detections,
        frame_idx: int,
        timestamp: float,
    ) -> DetectionResult:
        result = DetectionResult(frame_idx=frame_idx, timestamp=timestamp)

        if dets.class_id is None or len(dets) == 0:
            return result

        for i in range(len(dets)):
            cls_id = int(dets.class_id[i])
            conf = float(dets.confidence[i]) if dets.confidence is not None else 1.0
            x1, y1, x2, y2 = dets.xyxy[i].tolist()
            bbox = (x1, y1, x2, y2)

            if cls_id == _COCO_SPORTS_BALL:
                # Keep highest-confidence ball only
                det = Detection(ObjectClass.BALL, bbox, conf)
                if result.ball is None or conf > result.ball.confidence:
                    result.ball = det

            elif cls_id == _COCO_PERSON:
                # Simple heuristic: very short bounding boxes near edge = referee
                # This will be replaced by team-color clustering in a later sprint
                det = Detection(ObjectClass.PLAYER, bbox, conf)
                result.players.append(det)

        return result

    # ------------------------------------------------------------------
    # Visualisation helper (dev only)
    # ------------------------------------------------------------------

    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes on a copy of the frame. For debugging."""
        annotated = frame.copy()
        box_ann = sv.BoxAnnotator()
        label_ann = sv.LabelAnnotator()

        all_dets = result.all_objects
        if not all_dets:
            return annotated

        xyxy = np.array([d.bbox for d in all_dets])
        class_ids = np.array([int(d.obj_class) for d in all_dets])
        confidences = np.array([d.confidence for d in all_dets])
        tracker_ids = np.array([d.tracker_id for d in all_dets])

        sv_dets = sv.Detections(
            xyxy=xyxy,
            class_id=class_ids,
            confidence=confidences,
            tracker_id=tracker_ids,
        )

        labels = [
            f"#{d.tracker_id} {d.obj_class.name} {d.confidence:.2f}"
            for d in all_dets
        ]

        annotated = box_ann.annotate(scene=annotated, detections=sv_dets)
        annotated = label_ann.annotate(scene=annotated, detections=sv_dets, labels=labels)
        return annotated
