"""FastAPI routes for Explainable AI."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from fie.explainability.explainer import Explainer

router = APIRouter(prefix="/explain", tags=["Explainable AI"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GoalProbRequest(BaseModel):
    probability: float = Field(ge=0.0, le=1.0, description="Вероятность гола 0..1")
    language: str = Field(default="ru", description="'ru' или 'en'")
    compactness: float | None = None
    distance_to_goal: float | None = None
    players_between: int | None = None
    shot_angle_deg: float | None = None
    under_pressure: bool = False
    xg_context: float | None = None


class FatigueRequest(BaseModel):
    fatigue_score: float = Field(ge=0.0, le=100.0)
    language: str = "ru"
    player_id: int | None = None
    sprint_count: int | None = None
    speed_drop_pct: float | None = None
    minutes_played: float | None = None
    accel_count: int | None = None


class MistakeRequest(BaseModel):
    mistake_type: str = Field(description=(
        "Тип ошибки: defensive_gap / pressing_failure / "
        "late_recovery / open_corridor / line_break"
    ))
    language: str = "ru"
    player_id: int | None = None
    severity: str = "medium"
    zone: str | None = None
    minute: int | None = None
    context: str | None = None


class MatchPredictionRequest(BaseModel):
    home_win_prob: float = Field(ge=0.0, le=1.0)
    draw_prob: float = Field(ge=0.0, le=1.0)
    away_win_prob: float = Field(ge=0.0, le=1.0)
    language: str = "ru"
    home_team: str = "Home"
    away_team: str = "Away"
    key_factors: list[str] = Field(default_factory=list)
    home_xg: float | None = None
    away_xg: float | None = None
    home_fatigue_avg: float | None = None
    away_fatigue_avg: float | None = None


class TeamDNARequest(BaseModel):
    dna: dict = Field(description=(
        "DNA вектор: pressing_intensity, pressing_line, tempo, "
        "territory, attack_width, aggression, compactness"
    ))
    team_name: str = "Team"
    language: str = "ru"
    compare_with: dict | None = None
    compare_name: str = "Opponent"


class ScoutingRequest(BaseModel):
    talent_score: float = Field(ge=0.0, le=100.0)
    language: str = "ru"
    player_id: int | None = None
    position: str | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    readiness_pct: float | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/goal-probability")
async def explain_goal_prob(req: GoalProbRequest) -> dict:
    """
    Объяснить, почему вероятность гола именно такая.

    Пример: вероятность 38% → «потеряна компактность, открытый правый фланг,
    численное преимущество соперника 3v2».
    """
    e = Explainer(language=req.language).explain_goal_probability(
        req.probability,
        compactness=req.compactness,
        distance_to_goal=req.distance_to_goal,
        players_between=req.players_between,
        shot_angle_deg=req.shot_angle_deg,
        under_pressure=req.under_pressure,
        xg_context=req.xg_context,
    )
    return e.to_dict()


@router.post("/fatigue")
async def explain_fatigue(req: FatigueRequest) -> dict:
    """
    Объяснить, почему игрок в зоне риска усталости/травмы.

    Пример: «42 спринта — выше нормы, скорость упала на 18%,
    последние 15 мин без ускорений — рекомендуется замена».
    """
    e = Explainer(language=req.language).explain_fatigue(
        req.fatigue_score,
        player_id=req.player_id,
        sprint_count=req.sprint_count,
        speed_drop_pct=req.speed_drop_pct,
        minutes_played=req.minutes_played,
        accel_count=req.accel_count,
    )
    return e.to_dict()


@router.post("/tactical-mistake")
async def explain_mistake(req: MistakeRequest) -> dict:
    """
    Объяснить тактическую ошибку: что произошло, почему это проблема,
    что нужно исправить на тренировке.
    """
    e = Explainer(language=req.language).explain_tactical_mistake(
        req.mistake_type,
        player_id=req.player_id,
        severity=req.severity,
        zone=req.zone,
        minute=req.minute,
        context=req.context,
    )
    return e.to_dict()


@router.post("/match-prediction")
async def explain_match_prediction(req: MatchPredictionRequest) -> dict:
    """
    Объяснить предсказание исхода матча.

    Не просто «победа команды A — 64%», а полное объяснение
    на основе xG, усталости, тактических факторов.
    """
    e = Explainer(language=req.language).explain_match_prediction(
        req.home_win_prob, req.draw_prob, req.away_win_prob,
        home_team=req.home_team,
        away_team=req.away_team,
        key_factors=req.key_factors or [],
        home_xg=req.home_xg,
        away_xg=req.away_xg,
        home_fatigue_avg=req.home_fatigue_avg,
        away_fatigue_avg=req.away_fatigue_avg,
    )
    return e.to_dict()


@router.post("/team-dna")
async def explain_team_dna(req: TeamDNARequest) -> dict:
    """
    Объяснить тактический профиль команды (Team DNA).

    Переводит числовые векторы в понятные тактические характеристики:
    «Высокий прессинг, вертикальный темп, доминирование на правом фланге».
    """
    e = Explainer(language=req.language).explain_team_dna(
        req.dna,
        team_name=req.team_name,
        compare_with=req.compare_with,
        compare_name=req.compare_name,
    )
    return e.to_dict()


@router.post("/scouting")
async def explain_scouting(req: ScoutingRequest) -> dict:
    """
    Объяснить скаутинговую оценку игрока.

    Раскрывает, почему игрок получил такой talent_score,
    какие сильные/слабые стороны определили оценку.
    """
    e = Explainer(language=req.language).explain_scouting(
        req.talent_score,
        player_id=req.player_id,
        position=req.position,
        strengths=req.strengths or [],
        weaknesses=req.weaknesses or [],
        readiness_pct=req.readiness_pct,
    )
    return e.to_dict()
