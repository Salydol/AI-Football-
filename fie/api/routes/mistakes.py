"""FastAPI routes for Mistake Detection (Version 2)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.tactical.formation import FormationDetector
from fie.tactical.mistakes import MistakeDetector
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


class MistakeAnalysisRequest(BaseModel):
    frames: list[TrackingFrameInput] = Field(..., description="Трекинг-кадры матча")
    fps: float = Field(default=25.0)
    field_length: float = Field(default=105.0)
    field_width: float = Field(default=68.0)
    min_duration_frames: int = Field(
        default=8,
        description="Минимум кадров для фиксации ошибки (фильтр шума)",
    )
    use_formation: bool = Field(
        default=True,
        description="Использовать автоопределение формации для проверки линий",
    )


class MistakeAnalysisResponse(BaseModel):
    total_mistakes: int
    duration_seconds: float
    summary: dict
    mistakes_timeline: list[dict]   # все ошибки с временными метками


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/mistakes", response_model=MistakeAnalysisResponse)
async def analyze_mistakes(request: MistakeAnalysisRequest) -> MistakeAnalysisResponse:
    """
    Определить тактические ошибки из трекинг-данных.

    Типы ошибок:
    - **unmarked_opponent** — соперник без опеки в опасной зоне (HIGH)
    - **line_break** — игрок выбился из линии формации (MEDIUM/HIGH)
    - **wrong_zone** — игрок долго не в своей трети (LOW/MEDIUM)
    - **exposed_space** — дыра в обороне рядом с мячом (MEDIUM/HIGH)
    - **pressing_mismatch** — игрок не участвует в прессинге (LOW)
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

    # Инициализировать детекторы
    mistake_detector = MistakeDetector(
        field_length=request.field_length,
        field_width=request.field_width,
        fps=request.fps,
        min_duration=request.min_duration_frames,
    )

    formation_detector = FormationDetector(
        field_length=request.field_length,
        field_width=request.field_width,
    ) if request.use_formation else None

    # Обработать кадры
    for frame in tracking_frames:
        formation_left = None
        formation_right = None

        if formation_detector:
            formation_detector.update(frame.players)
            # Обновлять формацию каждые 25 кадров
            if frame.frame_idx % 25 == 0:
                formation_left, formation_right = formation_detector.detect_both_teams(
                    frame.players
                )

        mistake_detector.analyze(frame, formation_left, formation_right)

    # Результат
    summary = mistake_detector.get_summary()
    duration = len(tracking_frames) / request.fps

    logger.info(
        "Mistake detection complete | frames={} | mistakes={} | high={}",
        len(tracking_frames),
        summary.total_mistakes,
        summary.by_severity.get("high", 0),
    )

    return MistakeAnalysisResponse(
        total_mistakes=summary.total_mistakes,
        duration_seconds=round(duration, 1),
        summary={
            "by_type": summary.by_type,
            "by_severity": summary.by_severity,
            "by_player": {str(k): v for k, v in summary.by_player.items()},
            "worst_player_id": summary.worst_player_id,
            "most_common_type": summary.most_common_type,
        },
        mistakes_timeline=[m.to_dict() for m in summary.mistakes],
    )
