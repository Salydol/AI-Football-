"""FastAPI routes for Player Rating Engine."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.ratings.aggregator import MatchAggregator
from fie.ratings.calculator import Position, RatingCalculator
from fie.tracking.pipeline import TrackedBall, TrackedPlayer, TrackingFrame

router = APIRouter(prefix="/ratings", tags=["Player Ratings"])


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


class PlayerPositionInput(BaseModel):
    player_id: int
    position: str = Field(
        default="UNKNOWN",
        description="GK / DEF / MID / FWD / UNKNOWN",
    )


class PlayerRatingsRequest(BaseModel):
    frames: list[TrackingFrameInput] = Field(
        ..., description="Список трекинг-кадров матча"
    )
    fps: float = Field(default=25.0, description="Кадров в секунду видео")
    player_positions: list[PlayerPositionInput] = Field(
        default_factory=list,
        description="Позиции игроков (опционально). Если не указаны — UNKNOWN.",
    )


class PhysicalRatingOutput(BaseModel):
    speed: float
    acceleration: float
    endurance: float
    intensity: float
    overall: float


class TacticalRatingOutput(BaseModel):
    positioning: float
    pressing: float
    coverage: float
    overall: float


class PlayerRatingOutput(BaseModel):
    player_id: int
    position: str
    physical: PhysicalRatingOutput
    tactical: TacticalRatingOutput
    overall: float


class PlayerRatingsResponse(BaseModel):
    total_players: int
    match_duration_seconds: float
    ratings: list[PlayerRatingOutput]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/player-ratings", response_model=PlayerRatingsResponse)
async def player_ratings(request: PlayerRatingsRequest) -> PlayerRatingsResponse:
    """
    Рассчитать рейтинги игроков из трекинг-данных матча.

    Принимает список TrackingFrame (весь матч или его часть),
    агрегирует статистику и возвращает рейтинги 0–100 для каждого игрока.
    """
    if not request.frames:
        raise HTTPException(status_code=422, detail="frames list is empty")

    # --- Конвертировать входные данные в объекты TrackingFrame ---
    tracking_frames: list[TrackingFrame] = []
    for f in request.frames:
        ball = (
            TrackedBall(x=f.ball.x, y=f.ball.y, confidence=f.ball.confidence)
            if f.ball
            else None
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
        tracking_frames.append(
            TrackingFrame(
                frame_idx=f.frame_idx,
                timestamp=f.timestamp,
                players=players,
                ball=ball,
            )
        )

    # --- Агрегировать статистику ---
    agg = MatchAggregator(fps=request.fps)
    for frame in tracking_frames:
        agg.update(frame)

    all_stats = agg.get_all_stats()

    if not all_stats:
        raise HTTPException(status_code=422, detail="No players found in tracking data")

    # --- Позиции игроков ---
    positions = {
        p.player_id: Position(p.position.upper())
        for p in request.player_positions
        if p.position.upper() in Position._value2member_map_
    }

    # --- Рассчитать рейтинги ---
    calculator = RatingCalculator(
        match_duration_seconds=agg.match_duration_seconds
    )
    ratings = calculator.calculate_all(all_stats, positions)

    logger.info(
        "Ratings calculated for {} players | match duration: {:.0f}s",
        len(ratings),
        agg.match_duration_seconds,
    )

    return PlayerRatingsResponse(
        total_players=len(ratings),
        match_duration_seconds=round(agg.match_duration_seconds, 1),
        ratings=[
            PlayerRatingOutput(
                player_id=r.player_id,
                position=r.position.value,
                physical=PhysicalRatingOutput(**{
                    k: round(v, 1)
                    for k, v in {
                        "speed": r.physical.speed,
                        "acceleration": r.physical.acceleration,
                        "endurance": r.physical.endurance,
                        "intensity": r.physical.intensity,
                        "overall": r.physical.overall,
                    }.items()
                }),
                tactical=TacticalRatingOutput(**{
                    k: round(v, 1)
                    for k, v in {
                        "positioning": r.tactical.positioning,
                        "pressing": r.tactical.pressing,
                        "coverage": r.tactical.coverage,
                        "overall": r.tactical.overall,
                    }.items()
                }),
                overall=round(r.overall, 1),
            )
            for r in ratings
        ],
    )
