"""FastAPI routes for Academy Progress Tracker."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.analytics.academy import AcademySession, AcademyTracker, get_age_group

router = APIRouter(prefix="/academy", tags=["Academy"])

ACADEMY_DIR = Path("data/academy")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AddSessionRequest(BaseModel):
    player_id: int
    name: str = Field(default="")
    position: str = Field(default="UNKNOWN", description="GK/CB/FB/CDM/CM/CAM/WNG/ST")

    session_id: str
    date: str = Field(description="YYYY-MM-DD")
    session_type: str = Field(default="match", description="'match' или 'training'")
    age_at_session: int = Field(description="Возраст игрока на момент сессии")

    # Физика
    distance_km: float
    max_speed_kmh: float
    sprint_count: int
    high_accel_count: int = 0

    # Рейтинги
    physical_rating: float
    tactical_rating: float
    overall_rating: float

    # Дополнительно
    coach_note: str = ""
    minutes_played: float = 90.0
    fatigue_score: float = 0.0


class AcademyResponse(BaseModel):
    player_id: int
    name: str
    total_sessions: int
    profile: dict | None


# ---------------------------------------------------------------------------
# POST /academy/add-session
# ---------------------------------------------------------------------------

@router.post("/add-session", response_model=AcademyResponse)
async def add_session(request: AddSessionRequest) -> AcademyResponse:
    """
    Добавить сессию (матч/тренировку) в дневник академиста.

    Данные сохраняются в `data/academy/player_{id}.json`.
    При повторном вызове с тем же session_id данные обновляются.
    """
    tracker = AcademyTracker.load_or_create(
        player_id=request.player_id,
        name=request.name,
        position=request.position,
        save_dir=ACADEMY_DIR,
    )

    session = AcademySession(
        session_id=request.session_id,
        date=request.date,
        session_type=request.session_type,
        age_at_session=request.age_at_session,
        distance_km=request.distance_km,
        max_speed_kmh=request.max_speed_kmh,
        sprint_count=request.sprint_count,
        high_accel_count=request.high_accel_count,
        physical_rating=request.physical_rating,
        tactical_rating=request.tactical_rating,
        overall_rating=request.overall_rating,
        coach_note=request.coach_note,
        minutes_played=request.minutes_played,
        fatigue_score=request.fatigue_score,
    )

    tracker.add_session(session)
    tracker.save()
    profile = tracker.get_profile()

    age_group = get_age_group(request.age_at_session)
    logger.info(
        "Academy session added | player={} | age={} ({}) | overall={:.1f}",
        request.player_id, request.age_at_session, age_group.value, request.overall_rating,
    )

    return AcademyResponse(
        player_id=request.player_id,
        name=tracker.name,
        total_sessions=len(tracker._sessions),
        profile=profile.to_dict() if profile else None,
    )


# ---------------------------------------------------------------------------
# GET /academy/{player_id}/progress
# ---------------------------------------------------------------------------

@router.get("/{player_id}/progress", response_model=AcademyResponse)
async def get_progress(player_id: int) -> AcademyResponse:
    """Получить полный профиль прогресса академиста."""
    path = ACADEMY_DIR / f"player_{player_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Academy record not found: player {player_id}")

    tracker = AcademyTracker.load(path)
    profile = tracker.get_profile()

    return AcademyResponse(
        player_id=player_id,
        name=tracker.name,
        total_sessions=len(tracker._sessions),
        profile=profile.to_dict() if profile else None,
    )


# ---------------------------------------------------------------------------
# GET /academy/{player_id}/readiness
# ---------------------------------------------------------------------------

@router.get("/{player_id}/readiness")
async def get_readiness(player_id: int) -> dict:
    """
    Оценка готовности к переходу в основную команду.

    Возвращает:
    - `readiness_pct` — % готовности (0..100)
    - `readiness_label` — not_ready / developing / close / ready
    - `estimated_seasons` — сколько сезонов до готовности
    - `recommendations` — конкретные рекомендации
    """
    path = ACADEMY_DIR / f"player_{player_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Academy record not found: player {player_id}")

    tracker = AcademyTracker.load(path)
    profile = tracker.get_profile()

    if not profile:
        raise HTTPException(status_code=422, detail="Insufficient session data")

    return {
        "player_id": player_id,
        "name": tracker.name,
        "age": profile.age,
        "age_group": profile.age_group,
        "readiness": profile.first_team_readiness.to_dict(),
        "development_score": profile.development_score,
        "career_trend": profile.career_trend,
    }


# ---------------------------------------------------------------------------
# GET /academy/{player_id}/seasons
# ---------------------------------------------------------------------------

@router.get("/{player_id}/seasons")
async def get_seasons(player_id: int) -> dict:
    """История прогресса по сезонам."""
    path = ACADEMY_DIR / f"player_{player_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Academy record not found: player {player_id}")

    tracker = AcademyTracker.load(path)
    profile = tracker.get_profile()

    if not profile:
        raise HTTPException(status_code=422, detail="Insufficient session data")

    return {
        "player_id": player_id,
        "name": tracker.name,
        "seasons": [s.to_dict() for s in profile.season_progress],
        "career_trend": profile.career_trend,
    }


# ---------------------------------------------------------------------------
# GET /academy/list — все академисты
# ---------------------------------------------------------------------------

@router.get("/list")
async def list_academy_players() -> dict:
    """Список всех игроков академии с базовой информацией."""
    ACADEMY_DIR.mkdir(parents=True, exist_ok=True)
    players = []

    for path in sorted(ACADEMY_DIR.glob("player_*.json")):
        try:
            tracker = AcademyTracker.load(path)
            profile = tracker.get_profile()
            if profile:
                players.append({
                    "player_id": tracker.player_id,
                    "name": tracker.name,
                    "age": profile.age,
                    "age_group": profile.age_group,
                    "position": tracker.position,
                    "sessions": len(tracker._sessions),
                    "overall_rating": round(profile.avg_overall, 1),
                    "development_score": profile.development_score,
                    "readiness_pct": profile.first_team_readiness.readiness_pct,
                    "career_trend": profile.career_trend,
                })
        except Exception:
            continue

    players.sort(key=lambda p: p["readiness_pct"], reverse=True)
    return {"total": len(players), "players": players}
