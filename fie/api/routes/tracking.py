"""
FastAPI routes for the Tracking module.

POST /tracking/analyze-match
    Upload path to a video file → returns full tracking data for all frames.
    Heavy operation — in production this should be offloaded to a background task
    and results streamed via WebSocket or Kafka. For MVP it runs synchronously.

POST /tracking/analyze-match/stream
    Returns a streaming response (newline-delimited JSON) so the caller
    can process frames as they arrive instead of waiting for the full video.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from fie.tracking.calibration import IdentityCalibration, ManualCalibration
from fie.tracking.pipeline import TrackingFrame, TrackingPipeline
from fie.tracking.source import VideoFileSource, make_source
from fie.api.schemas.tracking import (
    AnalyzeMatchRequest,
    AnalyzeMatchResponse,
    PlayerTrackingData,
    TrackingFrameSchema,
    BallPosition,
)

router = APIRouter(prefix="/tracking", tags=["Tracking"])


def _build_calibration(request: AnalyzeMatchRequest):
    if not request.calibration_points or len(request.calibration_points) < 4:
        logger.warning("No calibration points provided — using IdentityCalibration (pixel coords)")
        return IdentityCalibration()

    return ManualCalibration(
        pixel_points=[(p.px, p.py) for p in request.calibration_points],
        field_points=[(p.fx, p.fy) for p in request.calibration_points],
    )


def _frame_to_schema(frame: TrackingFrame) -> TrackingFrameSchema:
    return TrackingFrameSchema(
        frame_idx=frame.frame_idx,
        timestamp=round(frame.timestamp, 3),
        players=[
            PlayerTrackingData(
                player_id=p.player_id,
                x=p.x,
                y=p.y,
                speed=p.speed,
                acceleration=p.acceleration,
                direction=p.direction,
            )
            for p in frame.players
        ],
        ball=(
            BallPosition(x=frame.ball.x, y=frame.ball.y, confidence=frame.ball.confidence)
            if frame.ball
            else None
        ),
    )


@router.post("/analyze-match", response_model=AnalyzeMatchResponse)
async def analyze_match(request: AnalyzeMatchRequest) -> AnalyzeMatchResponse:
    """
    Process a full match video and return tracking data for every frame.

    Warning: this blocks until the entire video is processed.
    Use /analyze-match/stream for large files.
    """
    calibration = _build_calibration(request)
    pipeline = TrackingPipeline(calibration=calibration)

    try:
        source = make_source(request.video_path, skip_frames=request.skip_frames)
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    frames: list[TrackingFrameSchema] = []
    last_timestamp = 0.0

    with source:
        for tracking_frame in pipeline.process(source):
            frames.append(_frame_to_schema(tracking_frame))
            last_timestamp = tracking_frame.timestamp

    return AnalyzeMatchResponse(
        total_frames=len(frames),
        duration_seconds=round(last_timestamp, 2),
        frames=frames,
    )


@router.post("/analyze-match/stream")
async def analyze_match_stream(request: AnalyzeMatchRequest) -> StreamingResponse:
    """
    Stream tracking frames as newline-delimited JSON (NDJSON).

    Each line is a JSON object representing one TrackingFrame.
    The client can parse and process frames incrementally.

    Example (curl):
        curl -X POST http://localhost:8000/tracking/analyze-match/stream \\
             -H 'Content-Type: application/json' \\
             -d '{"video_path": "/data/match.mp4"}' \\
             --no-buffer
    """
    calibration = _build_calibration(request)
    pipeline = TrackingPipeline(calibration=calibration)

    try:
        source = make_source(request.video_path, skip_frames=request.skip_frames)
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    def generate() -> Iterator[str]:
        with source:
            for tracking_frame in pipeline.process(source):
                schema = _frame_to_schema(tracking_frame)
                yield schema.model_dump_json() + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )
