"""FastAPI routes for Opponent Weakness Scanner."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.analytics.opponent import (
    OpponentMatch,
    OpponentScanner,
    OpponentWeaknessReport,
    ZoneStats,
)

router = APIRouter(prefix="/opponent", tags=["Opponent Scanner"])

REPORTS_DIR = Path("data/opponent_reports")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ZoneStatsSchema(BaseModel):
    zone: str
    losses: int = 0
    mistakes: int = 0
    goals_conceded: int = 0
    xg_conceded: float = 0.0
    pressing_failures: int = 0


class OpponentMatchSchema(BaseModel):
    match_id: str
    date: str
    opponent_name: str
    distance_km: float = 0.0
    sprint_count: int = 0
    max_speed_kmh: float = 0.0
    high_accel_count: int = 0
    pressing_intensity: float = 0.5
    pressing_line: float = 0.5
    compactness: float = 0.5
    territory: float = 0.5
    goals_scored: int = 0
    goals_conceded: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    defensive_mistakes: list[dict] = Field(default_factory=list)
    zone_stats: list[ZoneStatsSchema] = Field(default_factory=list)
    speed_by_period: dict[str, float] = Field(default_factory=dict)
    weak_players: dict[int, str] = Field(default_factory=dict)


class ScanRequest(BaseModel):
    opponent_name: str = Field(description="Название команды соперника")
    matches: list[OpponentMatchSchema] = Field(
        description="Список матчей соперника для анализа (минимум 1)"
    )
    save: bool = Field(default=True, description="Сохранить отчёт в файл")


# ---------------------------------------------------------------------------
# POST /opponent/scan
# ---------------------------------------------------------------------------

@router.post("/scan")
async def scan_opponent(request: ScanRequest) -> dict:
    """
    Анализировать слабости соперника по данным нескольких матчей.

    Принимает список матчей соперника и возвращает:
    - Слабые зоны поля (где допускает ошибки)
    - Периоды физического спада
    - Тактические уязвимости
    - Готовые рекомендации по атаке
    - Слабые игроки

    Чем больше матчей — тем точнее анализ. Рекомендуется 3-5 матчей.
    """
    if not request.matches:
        raise HTTPException(status_code=422, detail="Нужен хотя бы один матч")

    scanner = OpponentScanner(request.opponent_name)

    for m in request.matches:
        zone_stats_list = [
            ZoneStats(
                zone=z.zone,
                losses=z.losses,
                mistakes=z.mistakes,
                goals_conceded=z.goals_conceded,
                xg_conceded=z.xg_conceded,
                pressing_failures=z.pressing_failures,
            )
            for z in m.zone_stats
        ]
        scanner.add_match(OpponentMatch(
            match_id=m.match_id,
            date=m.date,
            opponent_name=m.opponent_name,
            distance_km=m.distance_km,
            sprint_count=m.sprint_count,
            max_speed_kmh=m.max_speed_kmh,
            high_accel_count=m.high_accel_count,
            pressing_intensity=m.pressing_intensity,
            pressing_line=m.pressing_line,
            compactness=m.compactness,
            territory=m.territory,
            goals_scored=m.goals_scored,
            goals_conceded=m.goals_conceded,
            xg_for=m.xg_for,
            xg_against=m.xg_against,
            defensive_mistakes=m.defensive_mistakes,
            zone_stats=zone_stats_list,
            speed_by_period=m.speed_by_period,
            weak_players=m.weak_players,
        ))

    report = scanner.analyze()
    logger.info(
        "Opponent scan: {} | {} matches | vuln_score={:.1f}",
        request.opponent_name, len(request.matches), report.overall_vulnerability_score,
    )

    saved_path = None
    if request.save:
        saved_path = str(report.save(REPORTS_DIR))

    result = report.to_dict()
    result["saved_path"] = saved_path
    return result


# ---------------------------------------------------------------------------
# GET /opponent/report/{opponent_name}
# ---------------------------------------------------------------------------

@router.get("/report/{opponent_name}")
async def get_report(opponent_name: str) -> dict:
    """Загрузить сохранённый отчёт по сопернику."""
    safe_name = opponent_name.replace(" ", "_").lower()
    path = REPORTS_DIR / f"{safe_name}_weakness.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Отчёт не найден: {opponent_name}. Сначала запустите POST /opponent/scan",
        )
    report = OpponentWeaknessReport.load(path)
    return report.to_dict()


# ---------------------------------------------------------------------------
# GET /opponent/list
# ---------------------------------------------------------------------------

@router.get("/list")
async def list_reports() -> dict:
    """Список всех сохранённых отчётов по соперникам."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(REPORTS_DIR.glob("*_weakness.json")):
        try:
            r = OpponentWeaknessReport.load(path)
            reports.append({
                "opponent_name": r.opponent_name,
                "matches_analyzed": r.matches_analyzed,
                "overall_vulnerability_score": r.overall_vulnerability_score,
                "generated_at": r.generated_at,
            })
        except Exception:
            continue
    return {"total": len(reports), "reports": reports}
