"""FastAPI routes for PDF Report generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from fie.ratings.aggregator import MatchAggregator
from fie.ratings.calculator import Position, RatingCalculator
from fie.reports.generator import MatchReportGenerator
from fie.tracking.pipeline import TrackedBall, TrackedPlayer, TrackingFrame

router = APIRouter(prefix="/reports", tags=["Reports"])


# ---------------------------------------------------------------------------
# Reuse input schemas (same as ratings route)
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
    position: str = Field(default="UNKNOWN", description="GK / DEF / MID / FWD / UNKNOWN")


class MatchReportRequest(BaseModel):
    frames: list[TrackingFrameInput] = Field(..., description="Трекинг-кадры матча")
    fps: float = Field(default=25.0, description="Кадров в секунду")
    player_positions: list[PlayerPositionInput] = Field(
        default_factory=list, description="Позиции игроков (опционально)"
    )
    match_title: str = Field(default="Match Report", description="Название матча")
    match_date: str | None = Field(default=None, description="Дата матча (DD Month YYYY)")
    home_team: str = Field(default="Home", description="Название команды хозяев")
    away_team: str = Field(default="Away", description="Название команды гостей")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/match-report",
    response_class=FileResponse,
    summary="Сгенерировать PDF-отчёт матча",
    responses={200: {"content": {"application/pdf": {}}, "description": "PDF report"}},
)
async def match_report(request: MatchReportRequest) -> FileResponse:
    """
    Принимает трекинг-кадры матча и возвращает PDF-отчёт.

    PDF содержит:
    - Обложку с названием матча
    - Сводку (общая статистика)
    - Таблицу рейтингов всех игроков
    - Детальную физическую статистику
    - Тепловые карты позиций (если установлен matplotlib)
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

    # Агрегация
    agg = MatchAggregator(fps=request.fps)
    for frame in tracking_frames:
        agg.update(frame)

    all_stats = agg.get_all_stats()
    if not all_stats:
        raise HTTPException(status_code=422, detail="No players found in tracking data")

    # Позиции
    positions = {
        p.player_id: Position(p.position.upper())
        for p in request.player_positions
        if p.position.upper() in Position._value2member_map_
    }

    # Рейтинги
    calculator = RatingCalculator(match_duration_seconds=agg.match_duration_seconds)
    ratings = calculator.calculate_all(all_stats, positions)

    # Генерация PDF во временный файл
    tmp_dir = Path(tempfile.mkdtemp())
    pdf_path = tmp_dir / "match_report.pdf"

    try:
        gen = MatchReportGenerator()
        gen.generate(
            output_path=pdf_path,
            ratings=ratings,
            aggregator=agg,
            match_title=request.match_title,
            match_date=request.match_date,
            home_team=request.home_team,
            away_team=request.away_team,
        )
    except Exception as exc:
        logger.exception("Failed to generate PDF: {}", exc)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc

    logger.info(
        "PDF report generated | players={} | duration={:.0f}s | size={:.1f}KB",
        len(ratings),
        agg.match_duration_seconds,
        pdf_path.stat().st_size / 1024,
    )

    filename = (
        request.match_title.replace(" ", "_").lower() + "_report.pdf"
    )
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
