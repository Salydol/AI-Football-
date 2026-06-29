"""FastAPI routes for Match Prediction and Goal Probability (Version 3)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.prediction.features import FeatureExtractor
from fie.prediction.goal_probability import GoalProbabilityEngine
from fie.prediction.match_predictor import MatchPredictor
from fie.tracking.pipeline import TrackedBall, TrackedPlayer, TrackingFrame

router = APIRouter(prefix="/prediction", tags=["Prediction"])


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


# ---------------------------------------------------------------------------
# Match Outcome Prediction
# ---------------------------------------------------------------------------

class MatchOutcomeRequest(BaseModel):
    frames: list[TrackingFrameInput] = Field(..., description="Трекинг-кадры матча")
    fps: float = Field(default=25.0)
    field_length: float = Field(default=105.0)
    field_width: float = Field(default=68.0)
    match_duration_seconds: float = Field(
        default=5400.0,
        description="Ожидаемая длительность матча для нормализации фич",
    )
    model_path: str | None = Field(
        default=None,
        description="Путь к обученной CatBoost модели (.cbm). Если не указан — rule-based.",
    )


class MatchOutcomeResponse(BaseModel):
    outcome: str
    probabilities: dict[str, float]
    confidence: float
    method: str
    features: dict


@router.post("/match-outcome", response_model=MatchOutcomeResponse)
async def predict_match_outcome(request: MatchOutcomeRequest) -> MatchOutcomeResponse:
    """
    Предсказать исход матча из трекинг-данных.

    Возвращает вероятности для трёх исходов:
    - **left_win** — победа команды с меньшим X-центром тяжести
    - **draw** — ничья
    - **right_win** — победа команды с большим X-центром тяжести

    Без обученной модели использует rule-based алгоритм
    на основе территориального преимущества, прессинга и физики.
    """
    if not request.frames:
        raise HTTPException(status_code=422, detail="frames list is empty")

    # Конвертировать кадры
    tracking_frames = _convert_frames(request.frames)

    # Извлечь фичи
    extractor = FeatureExtractor(
        field_length=request.field_length,
        field_width=request.field_width,
        fps=request.fps,
        match_duration=request.match_duration_seconds,
    )
    features = extractor.extract_from_frames(tracking_frames)

    # Предсказание
    predictor = MatchPredictor(model_path=request.model_path)
    result = predictor.predict(features)

    logger.info(
        "Match outcome prediction | outcome={} | confidence={:.2f} | method={}",
        result.outcome,
        result.confidence,
        result.method,
    )

    return MatchOutcomeResponse(
        outcome=result.outcome,
        probabilities=result.probabilities,
        confidence=result.confidence,
        method=result.method,
        features=features.to_dict(),
    )


# ---------------------------------------------------------------------------
# Goal Probability
# ---------------------------------------------------------------------------

class GoalProbRequest(BaseModel):
    frames: list[TrackingFrameInput] = Field(
        ..., description="Трекинг-кадры (последние N кадров для momentum)"
    )
    field_length: float = Field(default=105.0)
    field_width: float = Field(default=68.0)
    attacking_team: str = Field(
        default="both",
        description="'left', 'right' или 'both' для обеих команд",
    )


class GoalProbResponse(BaseModel):
    left_attack: dict | None = None    # xG для атаки левой команды
    right_attack: dict | None = None   # xG для атаки правой команды


@router.post("/goal-probability", response_model=GoalProbResponse)
async def goal_probability(request: GoalProbRequest) -> GoalProbResponse:
    """
    Вычислить xG (Expected Goals) и вероятность гола.

    Использует последний кадр из переданных для расчёта текущего xG.
    История кадров используется для momentum (растёт ли угроза).

    Возвращает:
    - **xg** — Expected Goals 0..1 (0.1 = 10% шанс гола с этой позиции)
    - **probability_5s** — вероятность гола в ближайшие 5 секунд
    - **probability_30s** — вероятность гола в ближайшие 30 секунд
    - **danger_zone** — мяч в зоне штрафной
    - **threat_level** — low / medium / high / critical
    """
    if not request.frames:
        raise HTTPException(status_code=422, detail="frames list is empty")

    tracking_frames = _convert_frames(request.frames)

    engine = GoalProbabilityEngine(
        field_length=request.field_length,
        field_width=request.field_width,
    )

    # Прокрутить историю через движок
    for frame in tracking_frames[:-1]:
        engine.compute(frame.ball, frame.players, "left")

    # Последний кадр — итоговый результат
    last = tracking_frames[-1]
    ball = last.ball
    players = last.players

    left_result = None
    right_result = None

    if request.attacking_team in ("left", "both"):
        r = engine.compute(ball, players, "left")
        left_result = r.to_dict() if r else None

    if request.attacking_team in ("right", "both"):
        r = engine.compute(ball, players, "right")
        right_result = r.to_dict() if r else None

    logger.info(
        "Goal probability | left_xg={} | right_xg={}",
        left_result.get("xg") if left_result else None,
        right_result.get("xg") if right_result else None,
    )

    return GoalProbResponse(
        left_attack=left_result,
        right_attack=right_result,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _convert_frames(frames_in: list[TrackingFrameInput]) -> list[TrackingFrame]:
    result = []
    for f in frames_in:
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
        result.append(TrackingFrame(
            frame_idx=f.frame_idx,
            timestamp=f.timestamp,
            players=players,
            ball=ball,
        ))
    return result
