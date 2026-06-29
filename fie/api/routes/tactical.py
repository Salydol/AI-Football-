"""FastAPI routes for Tactical Analysis (Version 2)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.tactical.pipeline import TacticalPipeline, TacticalSummary
from fie.tracking.pipeline import TrackedBall, TrackedPlayer, TrackingFrame

router = APIRouter(prefix="/tactical", tags=["Tactical Analysis"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TrackedPlayerInput(BaseModel):
    player_id: int
    x: float
    y: float
    speed: float
    acceleration: float
    direction: float


class TrackedBallInput(BaseModel):
    x: float
    y: float
    confidence: float


class TrackingFrameInput(BaseModel):
    frame_idx: int
    timestamp: float
    players: list[TrackedPlayerInput]
    ball: TrackedBallInput | None = None


class TacticalAnalysisRequest(BaseModel):
    frames: list[TrackingFrameInput] = Field(
        ..., description="Трекинг-кадры матча"
    )
    fps: float = Field(default=25.0, description="Кадров в секунду")
    field_length: float = Field(
        default=105.0,
        description="Длина поля в единицах координат (105 для метров, или пиксели)",
    )
    field_width: float = Field(
        default=68.0,
        description="Ширина поля",
    )
    formation_interval: int = Field(
        default=25,
        description="Каждые N кадров пересчитывать формацию",
    )
    include_frame_data: bool = Field(
        default=False,
        description="Включать ли данные каждого кадра в ответ (осторожно: большой JSON)",
    )


class TacticalAnalysisResponse(BaseModel):
    total_frames: int
    duration_seconds: float
    summary: dict
    frames: list[dict] | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=TacticalAnalysisResponse)
async def analyze_tactical(request: TacticalAnalysisRequest) -> TacticalAnalysisResponse:
    """
    Тактический анализ матча из трекинг-данных.

    Возвращает:
    - Доминирующую формацию обеих команд
    - Статистику прессинга (high press / mid block / deep block)
    - Среднюю компактность
    - Опционально: данные каждого кадра (formation + pressing + compactness)
    """
    if not request.frames:
        raise HTTPException(status_code=422, detail="frames list is empty")

    # Конвертировать кадры
    tracking_frames: list[TrackingFrame] = []
    for f in request.frames:
        ball = (
            TrackedBall(x=f.ball.x, y=f.ball.y, confidence=f.ball.confidence)
            if f.ball else None
        )
        players = [
            TrackedPlayer(
                player_id=p.player_id,
                x=p.x, y=p.y,
                speed=p.speed,
                acceleration=p.acceleration,
                direction=p.direction,
            )
            for p in f.players
        ]
        tracking_frames.append(TrackingFrame(
            frame_idx=f.frame_idx,
            timestamp=f.timestamp,
            players=players,
            ball=ball,
        ))

    # Запустить тактический pipeline
    pipeline = TacticalPipeline(
        field_length=request.field_length,
        field_width=request.field_width,
        formation_interval=request.formation_interval,
    )

    tactical_frames = pipeline.process_frames(tracking_frames)
    summary = pipeline.get_summary(fps=request.fps)

    logger.info(
        "Tactical analysis complete | frames={} | formation_left={} | formation_right={}",
        len(tactical_frames),
        summary.dominant_formation_left,
        summary.dominant_formation_right,
    )

    return TacticalAnalysisResponse(
        total_frames=summary.total_frames,
        duration_seconds=round(summary.duration_seconds, 1),
        summary=summary.to_dict(),
        frames=[f.to_dict() for f in tactical_frames] if request.include_frame_data else None,
    )


@router.post("/formation-only", response_model=dict)
async def detect_formation(request: TacticalAnalysisRequest) -> dict:
    """
    Только определение формации (быстрый эндпоинт).
    Возвращает доминирующую формацию и частоты за период.
    """
    if not request.frames:
        raise HTTPException(status_code=422, detail="frames list is empty")

    tracking_frames: list[TrackingFrame] = []
    for f in request.frames:
        ball = (
            TrackedBall(x=f.ball.x, y=f.ball.y, confidence=f.ball.confidence)
            if f.ball else None
        )
        players = [
            TrackedPlayer(
                player_id=p.player_id,
                x=p.x, y=p.y,
                speed=p.speed,
                acceleration=p.acceleration,
                direction=p.direction,
            )
            for p in f.players
        ]
        tracking_frames.append(TrackingFrame(
            frame_idx=f.frame_idx,
            timestamp=f.timestamp,
            players=players,
            ball=ball,
        ))

    pipeline = TacticalPipeline(
        field_length=request.field_length,
        field_width=request.field_width,
        formation_interval=request.formation_interval,
    )
    pipeline.process_frames(tracking_frames)
    summary = pipeline.get_summary(fps=request.fps)

    return {
        "dominant_formation_left": summary.dominant_formation_left,
        "dominant_formation_right": summary.dominant_formation_right,
        "formation_left_history": pipeline._formation_left_counts,
        "formation_right_history": pipeline._formation_right_counts,
    }
