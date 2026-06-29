"""FastAPI routes for LLM Coach Assistant and Match Story (Version 4)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from fie.llm.coach_assistant import CoachAssistant
from fie.llm.match_story import MatchStoryGenerator

router = APIRouter(prefix="/llm", tags=["LLM Analyst"])


# ---------------------------------------------------------------------------
# Shared schema для данных матча
# ---------------------------------------------------------------------------

class MatchSummaryInput(BaseModel):
    """Структурированные данные матча для LLM."""
    teams: str | None = None
    score: str | None = None
    duration_min: float | None = None

    physical: dict[str, Any] | None = Field(
        default=None,
        description="{'home': {distance_km, sprint_count, max_speed}, 'away': {...}}"
    )
    tactical: dict[str, Any] | None = Field(
        default=None,
        description="{'formation', 'pressing_style', 'pressing_intensity', 'territory_pct', 'compactness'}"
    )
    mistakes: list[dict[str, Any]] | None = Field(
        default=None,
        description="[{type, severity, frame_idx, player_id}]"
    )
    player_ratings: list[dict[str, Any]] | None = Field(
        default=None,
        description="[{player_id, overall, physical, tactical}]"
    )
    fatigue: dict[str, Any] | None = Field(
        default=None,
        description="{'critical_players': [], 'high_risk_players': [], 'team_fatigue_avg': 0}"
    )
    prediction: dict[str, Any] | None = Field(
        default=None,
        description="{'outcome': 'home_win', 'confidence': 0.72}"
    )


# ---------------------------------------------------------------------------
# POST /llm/coach-ask
# ---------------------------------------------------------------------------

class CoachAskRequest(BaseModel):
    question: str = Field(description="Вопрос тренера на любом языке")
    match_summary: MatchSummaryInput | None = None
    extra_context: str = Field(default="", description="Дополнительный контекст")
    anthropic_api_key: str | None = Field(
        default=None,
        description="Ключ Anthropic API (опционально, можно через env ANTHROPIC_API_KEY)"
    )
    openai_api_key: str | None = Field(
        default=None,
        description="Ключ OpenAI API (опционально)"
    )
    force_backend: str | None = Field(
        default=None,
        description="'anthropic' / 'openai' / 'rule_based'"
    )


class CoachAskResponse(BaseModel):
    answer: str
    backend: str
    tokens_used: int
    model: str


@router.post("/coach-ask", response_model=CoachAskResponse)
async def coach_ask(request: CoachAskRequest) -> CoachAskResponse:
    """
    Задать вопрос AI-ассистенту тренера.

    Примеры вопросов:
    - "Почему мы проигрываем в прессинге?"
    - "Кого заменить в первую очередь?"
    - "Как улучшить компактность блока?"
    - "What tactical adjustments should we make at half-time?"

    Без API ключа работает в rule-based режиме.
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question cannot be empty")

    assistant = CoachAssistant(
        anthropic_api_key=request.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=request.openai_api_key or os.getenv("OPENAI_API_KEY"),
        force_backend=request.force_backend,
    )

    summary = request.match_summary.model_dump(exclude_none=True) if request.match_summary else None

    response = assistant.ask(
        question=request.question,
        match_summary=summary,
        extra_context=request.extra_context,
    )

    logger.info(
        "Coach ask | backend={} | tokens={} | q='{}'",
        response.backend, response.tokens_used, request.question[:50],
    )

    return CoachAskResponse(
        answer=response.answer,
        backend=response.backend,
        tokens_used=response.tokens_used,
        model=response.model,
    )


# ---------------------------------------------------------------------------
# POST /llm/tactical-report
# ---------------------------------------------------------------------------

class TacticalReportRequest(BaseModel):
    match_summary: MatchSummaryInput
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


@router.post("/tactical-report", response_model=CoachAskResponse)
async def tactical_report(request: TacticalReportRequest) -> CoachAskResponse:
    """
    Сгенерировать полный тактический отчёт по матчу.

    Охватывает: общую оценку, прессинг, ошибки, лучших игроков, рекомендации.
    """
    assistant = CoachAssistant(
        anthropic_api_key=request.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=request.openai_api_key or os.getenv("OPENAI_API_KEY"),
    )

    summary = request.match_summary.model_dump(exclude_none=True)
    response = assistant.tactical_report(summary)

    return CoachAskResponse(
        answer=response.answer,
        backend=response.backend,
        tokens_used=response.tokens_used,
        model=response.model,
    )


# ---------------------------------------------------------------------------
# POST /llm/substitution-advice
# ---------------------------------------------------------------------------

@router.post("/substitution-advice", response_model=CoachAskResponse)
async def substitution_advice(request: TacticalReportRequest) -> CoachAskResponse:
    """
    Рекомендации по заменам на основе усталости и рейтингов.

    Возвращает приоритизированный список замен с обоснованием.
    """
    assistant = CoachAssistant(
        anthropic_api_key=request.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=request.openai_api_key or os.getenv("OPENAI_API_KEY"),
    )

    summary = request.match_summary.model_dump(exclude_none=True)
    response = assistant.substitution_advice(summary)

    return CoachAskResponse(
        answer=response.answer,
        backend=response.backend,
        tokens_used=response.tokens_used,
        model=response.model,
    )


# ---------------------------------------------------------------------------
# POST /llm/match-story
# ---------------------------------------------------------------------------

class MatchStoryRequest(BaseModel):
    match_summary: MatchSummaryInput
    language: str = Field(default="en", description="'en' или 'ru'")
    home_team: str = Field(default="Home")
    away_team: str = Field(default="Away")
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


class MatchStoryResponse(BaseModel):
    story: str
    language: str
    backend: str
    tokens_used: int


@router.post("/match-story", response_model=MatchStoryResponse)
async def match_story(request: MatchStoryRequest) -> MatchStoryResponse:
    """
    Сгенерировать нарратив матча в стиле спортивного репортажа.

    Включает: зачин, ключевые тактические битвы,
    лучшие выступления, критические ошибки, итоговый вывод.
    """
    assistant = CoachAssistant(
        anthropic_api_key=request.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=request.openai_api_key or os.getenv("OPENAI_API_KEY"),
    )
    generator = MatchStoryGenerator(assistant=assistant)

    summary = request.match_summary.model_dump(exclude_none=True)
    story = generator.generate(
        match_summary=summary,
        language=request.language,
        home_team=request.home_team,
        away_team=request.away_team,
    )

    logger.info(
        "Match story | lang={} | backend={} | len={}",
        story.language, story.backend, len(story.text),
    )

    return MatchStoryResponse(
        story=story.text,
        language=story.language,
        backend=story.backend,
        tokens_used=story.tokens_used,
    )


# ---------------------------------------------------------------------------
# GET /llm/status — проверить доступные бэкенды
# ---------------------------------------------------------------------------

@router.get("/status")
async def llm_status() -> dict:
    """Проверить какие LLM бэкенды доступны."""
    anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    openai_key = bool(os.getenv("OPENAI_API_KEY"))

    try:
        import anthropic  # noqa: F401
        anthropic_installed = True
    except ImportError:
        anthropic_installed = False

    try:
        import openai  # noqa: F401
        openai_installed = True
    except ImportError:
        openai_installed = False

    active_backend = "rule_based"
    if anthropic_key and anthropic_installed:
        active_backend = "anthropic (claude-haiku-4-5-20251001)"
    elif openai_key and openai_installed:
        active_backend = "openai (gpt-4o-mini)"

    return {
        "active_backend": active_backend,
        "anthropic": {
            "installed": anthropic_installed,
            "api_key_set": anthropic_key,
            "available": anthropic_key and anthropic_installed,
        },
        "openai": {
            "installed": openai_installed,
            "api_key_set": openai_key,
            "available": openai_key and openai_installed,
        },
        "rule_based": {"available": True},
    }
