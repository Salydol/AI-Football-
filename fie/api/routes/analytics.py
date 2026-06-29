"""FastAPI routes for Analytics — Fatigue Risk and Player Passport."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.analytics.fatigue import FatigueAnalyzer
from fie.analytics.passport import MatchRecord, PlayerPassport
from fie.tracking.pipeline import TrackedBall, TrackedPlayer, TrackingFrame

router = APIRouter(prefix="/analytics", tags=["Analytics"])

PASSPORT_DIR = Path("data/passports")


# ---------------------------------------------------------------------------
# Shared schemas
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


def _convert(frames: list[TrackingFrameInput]) -> list[TrackingFrame]:
    result = []
    for f in frames:
        ball = TrackedBall(x=f.ball.x, y=f.ball.y, confidence=f.ball.confidence) if f.ball else None
        players = [
            TrackedPlayer(player_id=p.player_id, x=p.x, y=p.y,
                          speed=p.speed, acceleration=p.acceleration, direction=p.direction)
            for p in f.players
        ]
        result.append(TrackingFrame(frame_idx=f.frame_idx, timestamp=f.timestamp,
                                    players=players, ball=ball))
    return result


# ---------------------------------------------------------------------------
# Fatigue Risk
# ---------------------------------------------------------------------------

class FatigueRequest(BaseModel):
    frames: list[TrackingFrameInput]
    fps: float = Field(default=25.0)
    field_length: float = Field(default=105.0)


class FatigueResponse(BaseModel):
    team_fatigue_avg: float
    critical_players: list[int]
    high_risk_players: list[int]
    players: list[dict]


@router.post("/fatigue-risk", response_model=FatigueResponse)
async def fatigue_risk(request: FatigueRequest) -> FatigueResponse:
    """
    Оценить риск травмы и усталости для всех игроков.

    Анализирует падение скорости и интенсивности относительно
    первых 15 минут матча.

    Уровни: **low** / **elevated** / **high** / **critical**
    """
    if not request.frames:
        raise HTTPException(status_code=422, detail="frames list is empty")

    analyzer = FatigueAnalyzer(fps=request.fps, field_length=request.field_length)
    summary = analyzer.process_frames(_convert(request.frames))

    logger.info(
        "Fatigue analysis | players={} | critical={} | avg_score={:.1f}",
        len(summary.players),
        len(summary.critical_players),
        summary.team_fatigue_avg,
    )

    return FatigueResponse(
        team_fatigue_avg=round(summary.team_fatigue_avg, 1),
        critical_players=summary.critical_players,
        high_risk_players=summary.high_risk_players,
        players=[p.to_dict() for p in summary.players],
    )


# ---------------------------------------------------------------------------
# Player Passport — add match
# ---------------------------------------------------------------------------

class AddMatchRequest(BaseModel):
    player_id: int
    position: str = Field(default="UNKNOWN", description="GK/CB/FB/CDM/CM/CAM/WNG/ST")
    match_id: str
    date: str = Field(description="YYYY-MM-DD")
    opponent: str = Field(default="Unknown")
    duration_minutes: float = Field(default=90.0)

    # Физика
    distance_km: float
    max_speed_kmh: float
    avg_speed_kmh: float
    sprint_count: int
    high_accel_count: int

    # Рейтинги
    physical_rating: float
    tactical_rating: float
    overall_rating: float

    # Тактика
    avg_x: float = 0.0
    avg_y: float = 0.0
    time_near_ball_s: float = 0.0

    # Усталость
    fatigue_score: float = 0.0
    injury_risk: float = 0.0


class PassportResponse(BaseModel):
    player_id: int
    total_matches: int
    profile: dict | None


@router.post("/passport/add-match", response_model=PassportResponse)
async def passport_add_match(request: AddMatchRequest) -> PassportResponse:
    """
    Добавить данные матча в паспорт игрока.

    Паспорт сохраняется в `data/passports/player_{id}.json`.
    При повторном вызове с тем же match_id данные обновляются.
    """
    passport = PlayerPassport.load_or_create(
        player_id=request.player_id,
        position=request.position,
        save_dir=PASSPORT_DIR,
    )

    record = MatchRecord(
        match_id=request.match_id,
        date=request.date,
        opponent=request.opponent,
        duration_minutes=request.duration_minutes,
        distance_km=request.distance_km,
        max_speed_kmh=request.max_speed_kmh,
        avg_speed_kmh=request.avg_speed_kmh,
        sprint_count=request.sprint_count,
        high_accel_count=request.high_accel_count,
        physical_rating=request.physical_rating,
        tactical_rating=request.tactical_rating,
        overall_rating=request.overall_rating,
        avg_x=request.avg_x,
        avg_y=request.avg_y,
        time_near_ball_s=request.time_near_ball_s,
        fatigue_score=request.fatigue_score,
        injury_risk=request.injury_risk,
    )

    passport.add_match(record)
    passport.save()
    profile = passport.get_profile()

    logger.info(
        "Passport updated | player={} | matches={} | trend={}",
        request.player_id,
        len(passport._matches),
        profile.trend if profile else "N/A",
    )

    return PassportResponse(
        player_id=request.player_id,
        total_matches=len(passport._matches),
        profile=profile.to_dict() if profile else None,
    )


@router.get("/passport/{player_id}", response_model=PassportResponse)
async def passport_get(player_id: int) -> PassportResponse:
    """Получить профиль игрока из паспорта."""
    path = PASSPORT_DIR / f"player_{player_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Passport not found for player {player_id}")

    passport = PlayerPassport.load(path)
    profile = passport.get_profile()

    return PassportResponse(
        player_id=player_id,
        total_matches=len(passport._matches),
        profile=profile.to_dict() if profile else None,
    )


@router.get("/passport/{player_id}/history")
async def passport_history(player_id: int) -> dict:
    """История матчей игрока."""
    path = PASSPORT_DIR / f"player_{player_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Passport not found for player {player_id}")

    passport = PlayerPassport.load(path)
    return {
        "player_id": player_id,
        "total_matches": len(passport._matches),
        "matches": [m.to_dict() for m in passport._matches],
    }
