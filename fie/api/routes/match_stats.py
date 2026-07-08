"""
Match Player Statistics API — секция 45 продуктовой спецификации.

Эндпоинты:
  GET  /matches/{match_id}/player-stats                        — все вкладки
  GET  /matches/{match_id}/player-stats/general
  GET  /matches/{match_id}/player-stats/attacking
  GET  /matches/{match_id}/player-stats/defending
  GET  /matches/{match_id}/player-stats/passing
  GET  /matches/{match_id}/player-stats/duels
  GET  /matches/{match_id}/player-stats/goalkeeping
  GET  /matches/{match_id}/lineups/performance-overlay         — оверлей по типу
  GET  /matches/{match_id}/players/{player_id}/match-card      — карточка игрока
  GET  /matches/{match_id}/players/{player_id}/ai-summary      — AI сводка
  POST /matches/{match_id}/players/compare                     — сравнение двух
  POST /matches/{match_id}/player-stats/export                 — экспорт CSV/JSON
  POST /matches/{match_id}/lineups/generate-ai-insights        — AI инсайты матча

Примечание: генерация данных базируется на rule-based движке из
fie.analytics.player_match_stats. В боевом режиме замените _get_match_roster()
вызовом к вашей БД / трекинговому пайплайну.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from fie.analytics.player_match_stats import (
    PlayerMatchStat,
    OVERLAY_TYPES,
    build_overlay,
    compare_players,
    generate_player_stats,
    rank_players,
    tab_ai_summary,
)

router = APIRouter(prefix="/matches", tags=["match-stats"])


# ---------------------------------------------------------------------------
# Helpers — fake roster (replace with DB query in production)
# ---------------------------------------------------------------------------

_SAMPLE_ROSTERS: dict[str, list[dict]] = {
    "default": [
        # team A
        dict(player_id="a1", player_name="М. Алиев",     team_id="teamA", shirt_number=1,  position="GK", minutes_played=90, is_starter=True),
        dict(player_id="a2", player_name="К. Орлов",     team_id="teamA", shirt_number=4,  position="CB", minutes_played=90, is_starter=True),
        dict(player_id="a3", player_name="Д. Нечаев",    team_id="teamA", shirt_number=5,  position="CB", minutes_played=90, is_starter=True),
        dict(player_id="a4", player_name="Р. Зверев",    team_id="teamA", shirt_number=3,  position="LB", minutes_played=90, is_starter=True),
        dict(player_id="a5", player_name="В. Ларин",     team_id="teamA", shirt_number=2,  position="RB", minutes_played=72, is_starter=True, substituted_out=72),
        dict(player_id="a6", player_name="Е. Соколов",   team_id="teamA", shirt_number=6,  position="CDM", minutes_played=90, is_starter=True),
        dict(player_id="a7", player_name="А. Кузьмин",   team_id="teamA", shirt_number=8,  position="CM", minutes_played=90, is_starter=True),
        dict(player_id="a8", player_name="М. Сидоров",   team_id="teamA", shirt_number=10, position="CAM", minutes_played=90, is_starter=True, is_captain=True, assists=1),
        dict(player_id="a9", player_name="И. Пирогов",   team_id="teamA", shirt_number=11, position="LW", minutes_played=90, is_starter=True),
        dict(player_id="a10", player_name="С. Фролов",   team_id="teamA", shirt_number=7,  position="RW", minutes_played=90, is_starter=True),
        dict(player_id="a11", player_name="Г. Козлов",   team_id="teamA", shirt_number=9,  position="ST", minutes_played=90, is_starter=True, goals=1),
        # team B
        dict(player_id="b1", player_name="О. Петров",    team_id="teamB", shirt_number=1,  position="GK", minutes_played=90, is_starter=True),
        dict(player_id="b2", player_name="Н. Быков",     team_id="teamB", shirt_number=4,  position="CB", minutes_played=90, is_starter=True),
        dict(player_id="b3", player_name="А. Симонов",   team_id="teamB", shirt_number=5,  position="CB", minutes_played=90, is_starter=True),
        dict(player_id="b4", player_name="Ю. Волков",    team_id="teamB", shirt_number=3,  position="LB", minutes_played=90, is_starter=True),
        dict(player_id="b5", player_name="Д. Крылов",    team_id="teamB", shirt_number=2,  position="RB", minutes_played=90, is_starter=True),
        dict(player_id="b6", player_name="Е. Морозов",   team_id="teamB", shirt_number=6,  position="CDM", minutes_played=90, is_starter=True, yellow_cards=1),
        dict(player_id="b7", player_name="С. Лебедев",   team_id="teamB", shirt_number=8,  position="CM", minutes_played=90, is_starter=True),
        dict(player_id="b8", player_name="Г. Тихонов",   team_id="teamB", shirt_number=10, position="CAM", minutes_played=90, is_starter=True, is_captain=True),
        dict(player_id="b9", player_name="К. Журавлёв",  team_id="teamB", shirt_number=11, position="LW", minutes_played=63, is_starter=True, substituted_out=63),
        dict(player_id="b10", player_name="В. Ильин",    team_id="teamB", shirt_number=7,  position="RW", minutes_played=90, is_starter=True),
        dict(player_id="b11", player_name="А. Захаров",  team_id="teamB", shirt_number=9,  position="ST", minutes_played=90, is_starter=True),
    ]
}

# Approximate pitch positions (x,y) for overlay — keyed by position label
_POSITION_XY: dict[str, tuple[float, float]] = {
    "GK": (5, 50), "CB": (20, 50), "LB": (20, 25), "RB": (20, 75),
    "CDM": (35, 50), "CM": (45, 50), "CAM": (60, 50),
    "LW": (70, 20), "RW": (70, 80), "ST": (80, 50),
}


def _get_match_roster(match_id: str) -> list[dict]:
    """Return roster dicts for given match_id (stub — replace with DB)."""
    return _SAMPLE_ROSTERS.get(match_id, _SAMPLE_ROSTERS["default"])


def _build_stats(match_id: str) -> list[PlayerMatchStat]:
    """Generate/retrieve full PlayerMatchStat list for the match."""
    roster = _get_match_roster(match_id)
    return [
        generate_player_stats(match_id=match_id, **p)
        for p in roster
    ]


_STAT_CACHE: dict[str, list[PlayerMatchStat]] = {}


def _stats(match_id: str) -> list[PlayerMatchStat]:
    if match_id not in _STAT_CACHE:
        _STAT_CACHE[match_id] = _build_stats(match_id)
    return _STAT_CACHE[match_id]


# ---------------------------------------------------------------------------
# Tab field subsets
# ---------------------------------------------------------------------------

_GENERAL_FIELDS = [
    "player_id", "player_name", "shirt_number", "position", "team_id",
    "minutes_played", "is_starter", "is_captain",
    "goals", "assists", "rating", "yellow_cards", "red_cards",
    "substituted_in", "substituted_out", "notes",
]
_ATTACKING_FIELDS = [
    "player_id", "player_name", "shirt_number", "position", "team_id",
    "shots_on_target", "shots_off_target", "shots_blocked", "total_shots",
    "xg", "xa", "xgot", "key_passes",
    "big_chances_missed", "big_chances_created", "hit_woodwork",
    "touches_in_box",
    "shot_impact_index", "finishing_quality_index",
    "chance_creation_index", "attacking_threat_score",
]
_DEFENDING_FIELDS = [
    "player_id", "player_name", "shirt_number", "position", "team_id",
    "tackles_won", "total_tackles", "interceptions", "clearances",
    "blocked_shots", "defensive_actions", "recoveries",
    "dribbled_past", "errors_leading_to_shot", "errors_leading_to_goal",
    "clearances_off_line",
    "defensive_reliability_index", "zone_protection_score",
]
_PASSING_FIELDS = [
    "player_id", "player_name", "shirt_number", "position", "team_id",
    "accurate_passes", "total_passes", "pass_completion_pct",
    "accurate_crosses", "total_crosses",
    "accurate_long_balls", "total_long_balls",
    "progressive_passes", "passes_into_final_third", "passes_into_box",
    "key_passes", "xa", "build_up_involvement",
    "build_up_quality_index",
]
_DUELS_FIELDS = [
    "player_id", "player_name", "shirt_number", "position", "team_id",
    "duels_won", "total_duels", "duel_win_pct",
    "ground_duels_won", "total_ground_duels",
    "aerial_duels_won", "total_aerial_duels",
    "successful_dribbles", "total_dribbles",
    "touches", "touches_in_final_third", "carries", "progressive_carries",
    "ball_retention", "press_resistance",
    "possession_lost", "fouls", "was_fouled", "offsides",
    "physical_dominance_index",
]
_GOALKEEPING_FIELDS = [
    "player_id", "player_name", "shirt_number", "position", "team_id",
    "goalkeeper_saves", "goals_prevented", "goals_conceded",
    "punches", "high_claims", "big_saves",
    "saves_from_inside_box", "saves_from_outside_box",
    "xgot_faced", "goal_kicks", "gk_pass_accuracy", "gk_long_balls",
    "sweeper_actions",
    "goalkeeper_impact_score", "save_difficulty_index",
]

_TAB_FIELDS = {
    "general": _GENERAL_FIELDS,
    "attacking": _ATTACKING_FIELDS,
    "defending": _DEFENDING_FIELDS,
    "passing": _PASSING_FIELDS,
    "duels": _DUELS_FIELDS,
    "goalkeeping": _GOALKEEPING_FIELDS,
}


def _subset(stat: PlayerMatchStat, fields: list[str]) -> dict:
    full = stat.to_dict()
    return {f: full[f] for f in fields if f in full}


def _tab_response(stats: list[PlayerMatchStat], tab: str) -> dict:
    fields = _TAB_FIELDS[tab]
    return {
        "tab": tab,
        "players": [_subset(s, fields) for s in stats],
        "ai_insight": tab_ai_summary(stats, tab),
    }


# ---------------------------------------------------------------------------
# Routes — Player Stats
# ---------------------------------------------------------------------------

@router.get("/{match_id}/player-stats")
async def get_all_player_stats(match_id: str) -> dict:
    """All tabs at once — section 33-38."""
    stats = _stats(match_id)
    return {
        "match_id": match_id,
        "tabs": {tab: _tab_response(stats, tab) for tab in _TAB_FIELDS},
        "rankings": rank_players(stats),
    }


@router.get("/{match_id}/player-stats/general")
async def get_general_stats(match_id: str) -> dict:
    return {"match_id": match_id, **_tab_response(_stats(match_id), "general")}


@router.get("/{match_id}/player-stats/attacking")
async def get_attacking_stats(match_id: str) -> dict:
    return {"match_id": match_id, **_tab_response(_stats(match_id), "attacking")}


@router.get("/{match_id}/player-stats/defending")
async def get_defending_stats(match_id: str) -> dict:
    return {"match_id": match_id, **_tab_response(_stats(match_id), "defending")}


@router.get("/{match_id}/player-stats/passing")
async def get_passing_stats(match_id: str) -> dict:
    return {"match_id": match_id, **_tab_response(_stats(match_id), "passing")}


@router.get("/{match_id}/player-stats/duels")
async def get_duels_stats(match_id: str) -> dict:
    return {"match_id": match_id, **_tab_response(_stats(match_id), "duels")}


@router.get("/{match_id}/player-stats/goalkeeping")
async def get_goalkeeping_stats(match_id: str) -> dict:
    gk_stats = _stats(match_id)  # include all; frontend filters
    return {"match_id": match_id, **_tab_response(gk_stats, "goalkeeping")}


# ---------------------------------------------------------------------------
# Routes — Performance Overlay (section 40)
# ---------------------------------------------------------------------------

@router.get("/{match_id}/lineups/performance-overlay")
async def get_performance_overlay(
    match_id: str,
    type: Literal["rating", "shooting", "passing", "dribbling", "defending"] = Query(
        "rating", description="Тип оверлея"
    ),
) -> dict:
    """Overlay data с координатами на поле — section 40."""
    stats = _stats(match_id)
    overlays = []
    for stat in stats:
        xy = _POSITION_XY.get(stat.position, (50, 50))
        # Spread within position group by shirt number
        offset_x = (stat.shirt_number % 3 - 1) * 2.5
        offset_y = (stat.shirt_number % 5 - 2) * 3.0
        overlays.append(
            build_overlay(stat, type, xy[0] + offset_x, xy[1] + offset_y)
        )
    return {
        "match_id": match_id,
        "overlay_type": type,
        "players": overlays,
    }


# ---------------------------------------------------------------------------
# Routes — Player Match Card (section 39)
# ---------------------------------------------------------------------------

@router.get("/{match_id}/players/{player_id}/match-card")
async def get_player_match_card(match_id: str, player_id: str) -> dict:
    """Compact card: rating, goals, assists, key moments — section 39."""
    stat = next((s for s in _stats(match_id) if s.player_id == player_id), None)
    if not stat:
        raise HTTPException(404, f"Player {player_id} not found in match {match_id}")

    return {
        "match_id": match_id,
        "player_id": stat.player_id,
        "player_name": stat.player_name,
        "shirt_number": stat.shirt_number,
        "position": stat.position,
        "team_id": stat.team_id,
        "is_starter": stat.is_starter,
        "minutes_played": stat.minutes_played,
        "is_captain": stat.is_captain,
        "substituted_in": stat.substituted_in,
        "substituted_out": stat.substituted_out,
        # Summary stats
        "rating": stat.rating,
        "goals": stat.goals,
        "assists": stat.assists,
        "yellow_cards": stat.yellow_cards,
        "red_cards": stat.red_cards,
        # Key attacking
        "xg": stat.xg,
        "xa": stat.xa,
        "shots_on_target": stat.shots_on_target,
        "total_shots": stat.total_shots,
        "key_passes": stat.key_passes,
        "big_chances_missed": stat.big_chances_missed,
        # Key defensive
        "tackles_won": stat.tackles_won,
        "interceptions": stat.interceptions,
        "clearances": stat.clearances,
        # Pass
        "pass_completion_pct": stat.pass_completion_pct,
        # Indices
        "shot_impact_index": stat.shot_impact_index,
        "chance_creation_index": stat.chance_creation_index,
        "defensive_reliability_index": stat.defensive_reliability_index,
        "build_up_quality_index": stat.build_up_quality_index,
        "physical_dominance_index": stat.physical_dominance_index,
        # GK
        "goalkeeper_saves": stat.goalkeeper_saves if stat.position == "GK" else None,
        "goals_conceded": stat.goals_conceded if stat.position == "GK" else None,
        # Notes
        "notes": stat.notes,
        "ai_summary": stat.ai_summary,
    }


# ---------------------------------------------------------------------------
# Routes — AI Summary (section 42)
# ---------------------------------------------------------------------------

@router.get("/{match_id}/players/{player_id}/ai-summary")
async def get_player_ai_summary(match_id: str, player_id: str) -> dict:
    """Detailed AI-generated player narrative — section 42."""
    stat = next((s for s in _stats(match_id) if s.player_id == player_id), None)
    if not stat:
        raise HTTPException(404, f"Player {player_id} not found in match {match_id}")
    return {
        "match_id": match_id,
        "player_id": stat.player_id,
        "player_name": stat.player_name,
        "rating": stat.rating,
        "notes": stat.notes,
        "ai_summary": stat.ai_summary,
    }


# ---------------------------------------------------------------------------
# Routes — Comparison (section 43)
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    player_a_id: str
    player_b_id: str


@router.post("/{match_id}/players/compare")
async def compare_two_players(match_id: str, body: CompareRequest) -> dict:
    """Side-by-side comparison — section 43."""
    stats = _stats(match_id)
    a = next((s for s in stats if s.player_id == body.player_a_id), None)
    b = next((s for s in stats if s.player_id == body.player_b_id), None)
    if not a:
        raise HTTPException(404, f"Player {body.player_a_id} not in match")
    if not b:
        raise HTTPException(404, f"Player {body.player_b_id} not in match")
    return {"match_id": match_id, **compare_players(a, b)}


# ---------------------------------------------------------------------------
# Routes — Export (section 45)
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    format: Literal["json", "csv"] = "json"
    tab: Literal["general", "attacking", "defending", "passing", "duels", "goalkeeping", "all"] = "all"
    team_id: Optional[str] = None


@router.post("/{match_id}/player-stats/export")
async def export_player_stats(match_id: str, body: ExportRequest):
    """Export stats as JSON or CSV — section 45."""
    stats = _stats(match_id)
    if body.team_id:
        stats = [s for s in stats if s.team_id == body.team_id]
    if not stats:
        raise HTTPException(404, "No stats found")

    if body.tab == "all":
        rows = [s.to_dict() for s in stats]
    else:
        fields = _TAB_FIELDS[body.tab]
        rows = [_subset(s, fields) for s in stats]

    if body.format == "json":
        return JSONResponse(content={"match_id": match_id, "tab": body.tab, "data": rows})

    # CSV
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=match_{match_id}_{body.tab}.csv"},
    )


# ---------------------------------------------------------------------------
# Routes — AI Insights (section 48)
# ---------------------------------------------------------------------------

@router.post("/{match_id}/lineups/generate-ai-insights")
async def generate_ai_insights(match_id: str) -> dict:
    """Generate full match AI insights across all tabs — section 48."""
    stats = _stats(match_id)
    teams = list({s.team_id for s in stats})

    insights: dict = {
        "match_id": match_id,
        "teams": teams,
        "rankings": rank_players(stats),
        "tab_insights": {tab: tab_ai_summary(stats, tab) for tab in _TAB_FIELDS},
        "team_insights": {},
    }

    for team_id in teams:
        team_stats = [s for s in stats if s.team_id == team_id]
        insights["team_insights"][team_id] = {
            "avg_rating": round(sum(s.rating for s in team_stats) / len(team_stats), 2),
            "total_goals": sum(s.goals for s in team_stats),
            "total_assists": sum(s.assists for s in team_stats),
            "total_xg": round(sum(s.xg for s in team_stats), 2),
            "total_passes": sum(s.total_passes for s in team_stats),
            "pass_completion_pct": round(
                sum(s.accurate_passes for s in team_stats) /
                max(sum(s.total_passes for s in team_stats), 1) * 100, 1
            ),
            "total_defensive_actions": sum(s.defensive_actions for s in team_stats),
            "best_player": max(team_stats, key=lambda s: s.rating).player_name,
        }

    return insights
