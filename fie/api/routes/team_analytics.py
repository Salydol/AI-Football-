"""FastAPI routes for Team DNA and Scouting Radar."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.analytics.team_dna import TeamDNA, TeamDNAVector
from fie.analytics.scouting import PlayerSimilarityEngine, TargetProfile, build_radar_from_passport
from fie.tracking.pipeline import TrackedBall, TrackedPlayer, TrackingFrame

router = APIRouter(prefix="/analytics", tags=["Team Analytics"])

DNA_DIR = Path("data/team_dna")
PASSPORT_DIR = Path("data/passports")


# ---------------------------------------------------------------------------
# Shared input schemas
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
# Team DNA — добавить матч
# ---------------------------------------------------------------------------

class TeamDNAMatchRequest(BaseModel):
    team_id: str
    team_name: str = Field(default="")
    team_side: str = Field(default="left", description="'left' или 'right' — какая сторона наша команда")
    frames: list[TrackingFrameInput]
    field_length: float = Field(default=105.0)
    field_width: float = Field(default=68.0)


class TeamDNAResponse(BaseModel):
    team_id: str
    team_name: str
    total_matches: int
    profile: dict | None


@router.post("/team-dna/add-match", response_model=TeamDNAResponse)
async def team_dna_add_match(request: TeamDNAMatchRequest) -> TeamDNAResponse:
    """
    Добавить матч в ДНК команды.

    ДНК накапливается за несколько матчей и отражает стиль игры:
    прессинг, темп, территориальное доминирование, ширина атаки и т.д.

    Файл сохраняется в `data/team_dna/{team_id}.json`.
    """
    if not request.frames:
        raise HTTPException(status_code=422, detail="frames list is empty")

    team = TeamDNA.load_or_create(
        team_id=request.team_id,
        team_name=request.team_name,
        save_dir=DNA_DIR,
    )

    dna_vec = team.add_match(
        frames=_convert(request.frames),
        team=request.team_side,
        field_length=request.field_length,
        field_width=request.field_width,
    )
    team.save()
    profile = team.get_profile()

    logger.info(
        "Team DNA updated | team={} | matches={} | pressing={:.2f} | tempo={:.2f}",
        request.team_id,
        len(team._match_dnas),
        dna_vec.pressing_intensity,
        dna_vec.tempo,
    )

    return TeamDNAResponse(
        team_id=request.team_id,
        team_name=team.team_name,
        total_matches=len(team._match_dnas),
        profile=profile.to_dict() if profile else None,
    )


@router.get("/team-dna/{team_id}", response_model=TeamDNAResponse)
async def team_dna_get(team_id: str) -> TeamDNAResponse:
    """Получить ДНК-профиль команды."""
    path = DNA_DIR / f"{team_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Team DNA not found: {team_id}")
    team = TeamDNA.load(path)
    profile = team.get_profile()
    return TeamDNAResponse(
        team_id=team_id,
        team_name=team.team_name,
        total_matches=len(team._match_dnas),
        profile=profile.to_dict() if profile else None,
    )


@router.get("/team-dna/{team_id}/compare/{team_id_2}")
async def team_dna_compare(team_id: str, team_id_2: str) -> dict:
    """Сравнить ДНК двух команд."""
    for tid in [team_id, team_id_2]:
        if not (DNA_DIR / f"{tid}.json").exists():
            raise HTTPException(status_code=404, detail=f"Team DNA not found: {tid}")
    t1 = TeamDNA.load(DNA_DIR / f"{team_id}.json")
    t2 = TeamDNA.load(DNA_DIR / f"{team_id_2}.json")
    return t1.compare(t2)


# ---------------------------------------------------------------------------
# Scouting Radar
# ---------------------------------------------------------------------------

@router.get("/scouting/radar/{player_id}")
async def scouting_radar(player_id: int) -> dict:
    """Получить радар-профиль игрока из его паспорта."""
    path = PASSPORT_DIR / f"player_{player_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Passport not found: player {player_id}")

    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    matches = raw.get("matches", [])
    if not matches:
        raise HTTPException(status_code=422, detail="No match data in passport")

    import numpy as np
    avg_speed = float(np.mean([m.get("max_speed_kmh", 0) for m in matches]))
    avg_dist = float(np.mean([m.get("distance_km", 0) for m in matches]))
    avg_sprint = float(np.mean([m.get("sprint_count", 0) for m in matches]))
    avg_rating = float(np.mean([m.get("overall_rating", 0) for m in matches]))
    talent = min(avg_rating * 0.6 + min(avg_speed / 35.0, 1.0) * 10, 100.0)

    profile_data = {
        "player_id": player_id,
        "position": raw.get("position", "UNKNOWN"),
        "avg_max_speed": avg_speed,
        "avg_distance_km": avg_dist,
        "avg_sprint_count": avg_sprint,
        "avg_overall_rating": avg_rating,
        "talent_index": talent,
        "match_history": matches,
    }

    radar = build_radar_from_passport(profile_data)
    if not radar:
        raise HTTPException(status_code=422, detail="Could not build radar from passport data")

    return radar.to_dict()


class FindSimilarRequest(BaseModel):
    player_id: int
    top_k: int = Field(default=5, ge=1, le=20)
    position_filter: str | None = Field(default=None, description="GK/CB/FB/CDM/CM/CAM/WNG/ST")


@router.post("/scouting/find-similar")
async def scouting_find_similar(request: FindSimilarRequest) -> dict:
    """
    Найти игроков похожих по стилю на указанного.

    Сравнивает по 8-метричному радар-вектору.
    Возвращает top_k наиболее похожих игроков.
    """
    engine = PlayerSimilarityEngine(save_dir=PASSPORT_DIR)
    results = engine.find_similar(
        player_id=request.player_id,
        top_k=request.top_k,
        position_filter=request.position_filter,
    )
    return {
        "player_id": request.player_id,
        "top_k": request.top_k,
        "results": results,
    }


class ScoutingTargetRequest(BaseModel):
    pace_min: float = 0.0
    pace_weight: float = 0.0
    stamina_min: float = 0.0
    stamina_weight: float = 0.0
    sprint_power_min: float = 0.0
    sprint_power_weight: float = 0.0
    pressing_min: float = 0.0
    pressing_weight: float = 0.0
    positioning_min: float = 0.0
    positioning_weight: float = 0.0
    consistency_min: float = 0.0
    consistency_weight: float = 0.0
    form_min: float = 0.0
    form_weight: float = 0.0
    potential_min: float = 0.0
    potential_weight: float = 0.0
    position_filter: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


@router.post("/scouting/search")
async def scouting_search(request: ScoutingTargetRequest) -> dict:
    """
    Поиск игроков под целевой профиль.

    Пример — ищем быстрого вингера:
    ```json
    {
      "pace_min": 70, "pace_weight": 1.0,
      "sprint_power_min": 60, "sprint_power_weight": 0.8,
      "position_filter": "WNG",
      "top_k": 5
    }
    ```
    """
    engine = PlayerSimilarityEngine(save_dir=PASSPORT_DIR)
    target = TargetProfile(
        pace_min=request.pace_min,
        pace_weight=request.pace_weight,
        stamina_min=request.stamina_min,
        stamina_weight=request.stamina_weight,
        sprint_power_min=request.sprint_power_min,
        sprint_power_weight=request.sprint_power_weight,
        pressing_min=request.pressing_min,
        pressing_weight=request.pressing_weight,
        positioning_min=request.positioning_min,
        positioning_weight=request.positioning_weight,
        consistency_min=request.consistency_min,
        consistency_weight=request.consistency_weight,
        form_min=request.form_min,
        form_weight=request.form_weight,
        potential_min=request.potential_min,
        potential_weight=request.potential_weight,
        position_filter=request.position_filter,
    )
    results = engine.find_by_target(target, top_k=request.top_k)
    return {
        "total_found": len(results),
        "results": [r.to_dict() for r in results],
    }


@router.get("/scouting/compare/{player_id_1}/{player_id_2}")
async def scouting_compare(player_id_1: int, player_id_2: int) -> dict:
    """Сравнить двух игроков по радар-метрикам."""
    engine = PlayerSimilarityEngine(save_dir=PASSPORT_DIR)
    return engine.compare_players(player_id_1, player_id_2)
