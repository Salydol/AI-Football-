"""
Field calibration — homography from camera pixels to pitch metres.

The football pitch is modelled as a 2D plane:
  - Origin (0, 0) = top-left corner of the pitch
  - X axis → along the length (0..105 m)
  - Y axis → along the width  (0..68 m)

Two calibration modes:

1. ManualCalibration
   You manually click (or hard-code) 4+ corresponding points:
     pixel (px, py) ↔ field (fx, fy)
   Then OpenCV computes the homography matrix H.

2. LineCalibration (TODO — next sprint)
   Detects pitch lines automatically via Hough transforms.
   Works without manual annotation.

Usage:
    cal = ManualCalibration(
        pixel_points=[(100, 400), (1800, 400), (1800, 900), (100, 900)],
        field_points=[(0, 0),    (105, 0),    (105, 68),   (0, 68)],
    )
    x_m, y_m = cal.to_field(px=950, py=650)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from fie.config import settings


@dataclass(frozen=True)
class FieldPoint:
    """A point in pitch-metric space."""
    x: float  # metres from left touchline
    y: float  # metres from top goal line


class FieldCalibration(ABC):
    """Abstract calibration — maps pixel coords to field metres."""

    @abstractmethod
    def to_field(self, px: float, py: float) -> FieldPoint | None:
        """Convert pixel (px, py) to field metres. Returns None if outside pitch."""
        ...

    @abstractmethod
    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Convert field (x, y) metres back to pixel coords."""
        ...

    def is_on_pitch(self, point: FieldPoint, margin: float = 2.0) -> bool:
        """Check whether a field point lies within pitch boundaries (+ margin)."""
        return (
            -margin <= point.x <= settings.field_length_m + margin
            and -margin <= point.y <= settings.field_width_m + margin
        )


class ManualCalibration(FieldCalibration):
    """
    Homography calibration from manually provided point correspondences.

    At minimum 4 non-collinear points are required.
    More points = better accuracy (OpenCV solves least-squares).

    Typical landmarks to use:
      - 4 corner flags
      - penalty spots
      - centre circle centre
      - penalty area corners
    """

    def __init__(
        self,
        pixel_points: list[tuple[float, float]],
        field_points: list[tuple[float, float]],
    ) -> None:
        if len(pixel_points) < 4 or len(pixel_points) != len(field_points):
            raise ValueError(
                f"Need at least 4 matching point pairs, got {len(pixel_points)} px "
                f"and {len(field_points)} field"
            )

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(field_points, dtype=np.float32)

        # RANSAC tolerates a few bad manual annotations
        self._H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=2.0)
        self._H_inv, _ = cv2.findHomography(dst, src, cv2.RANSAC, ransacReprojThreshold=2.0)

        inliers = int(mask.sum()) if mask is not None else len(pixel_points)
        total = len(pixel_points)
        if inliers < total:
            import warnings
            warnings.warn(
                f"Homography: {total - inliers}/{total} point(s) rejected as outliers",
                stacklevel=2,
            )

    def to_field(self, px: float, py: float) -> FieldPoint | None:
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)[0][0]
        fp = FieldPoint(x=float(result[0]), y=float(result[1]))
        return fp if self.is_on_pitch(fp) else None

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H_inv)[0][0]
        return (float(result[0]), float(result[1]))

    @classmethod
    def from_full_frame(
        cls,
        frame_width: int,
        frame_height: int,
        *,
        pitch_length: float | None = None,
        pitch_width: float | None = None,
    ) -> "ManualCalibration":
        """
        Quick-start calibration: assumes the video frame shows the entire
        pitch and maps the 4 frame corners to the 4 pitch corners.

        This is inaccurate for real broadcasts (camera angle, zoom, etc.)
        but useful for top-down drone footage or synthetic data.
        """
        pl = pitch_length or settings.field_length_m
        pw = pitch_width or settings.field_width_m
        w, h = frame_width, frame_height

        return cls(
            pixel_points=[
                (0, 0), (w, 0), (w, h), (0, h),
            ],
            field_points=[
                (0, 0), (pl, 0), (pl, pw), (0, pw),
            ],
        )


class IdentityCalibration(FieldCalibration):
    """
    No-op calibration — returns pixel coords as 'field' coords.

    Use when you don't have homography data yet but want
    the pipeline to run end-to-end.
    """

    def to_field(self, px: float, py: float) -> FieldPoint:
        return FieldPoint(x=px, y=py)

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (x, y)

    def is_on_pitch(self, point: FieldPoint, margin: float = 2.0) -> bool:
        return True
