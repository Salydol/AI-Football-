"""Pydantic schemas for the tracking API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class CalibrationPoint(BaseModel):
    px: float = Field(..., description="Pixel X")
    py: float = Field(..., description="Pixel Y")
    fx: float = Field(..., description="Field X in metres")
    fy: float = Field(..., description="Field Y in metres")


class AnalyzeMatchRequest(BaseModel):
    video_path: str = Field(
        ...,
        description="Absolute path to the video file, or rtsp:// URL",
        examples=["/data/match.mp4", "rtsp://camera.local:554/stream"],
    )
    calibration_points: list[CalibrationPoint] | None = Field(
        default=None,
        description=(
            "Optional list of pixel↔field correspondences for homography calibration. "
            "Minimum 4 points. If omitted, IdentityCalibration is used (pixel coords)."
        ),
    )
    skip_frames: int = Field(
        default=0,
        ge=0,
        description="Process every N+1 frames. 0 = every frame, 1 = every other frame.",
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class BallPosition(BaseModel):
    x: float
    y: float
    confidence: float


class PlayerTrackingData(BaseModel):
    player_id: int
    x: float = Field(..., description="Metres from left touchline")
    y: float = Field(..., description="Metres from top goal line")
    speed: float = Field(..., description="Speed in km/h")
    acceleration: float = Field(..., description="Acceleration in m/s²")
    direction: float = Field(..., description="Direction in degrees (0=right, 90=up)")


class TrackingFrameSchema(BaseModel):
    frame_idx: int
    timestamp: float = Field(..., description="Seconds from match start")
    players: list[PlayerTrackingData]
    ball: BallPosition | None


class AnalyzeMatchResponse(BaseModel):
    total_frames: int
    duration_seconds: float
    frames: list[TrackingFrameSchema]


# ---------------------------------------------------------------------------
# Streaming / single-frame
# ---------------------------------------------------------------------------

class TrackFrameRequest(BaseModel):
    """For single-frame analysis (used in streaming scenarios)."""
    frame_idx: int
    timestamp: float
    # Frame is passed as raw bytes via multipart upload


class TrackFrameResponse(TrackingFrameSchema):
    pass
