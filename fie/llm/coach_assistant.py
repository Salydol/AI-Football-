"""
Coach Assistant — LLM-based тактический ассистент тренера.

Принимает данные матча в структурированном виде и генерирует:
  - Тактический разбор
  - Ответы на вопросы тренера
  - Рекомендации по заменам
  - Анализ ошибок

Бэкенды (в порядке приоритета):
  1. Anthropic Claude API (ANTHROPIC_API_KEY в env или config)
  2. OpenAI API (OPENAI_API_KEY в env)
  3. Ollama (локально, бесплатно — OLLAMA_MODEL в env, default: llama3)
  4. Rule-based fallback (работает без API ключей, 100% offline)

Использование:
    assistant = CoachAssistant()
    response = assistant.ask(
        question="Почему мы проигрываем в прессинге?",
        match_summary=summary_dict,
    )
    print(response.answer)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class CoachResponse:
    answer: str
    backend: str        # "anthropic" / "openai" / "rule_based"
    tokens_used: int = 0
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "backend": self.backend,
            "tokens_used": self.tokens_used,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an elite football (soccer) tactical analyst and coach assistant.
You analyze match data and provide concise, actionable insights.
You speak directly to the coaching staff. Be specific, reference actual numbers from the data.
Respond in the same language as the question (English or Russian).
Keep answers focused and under 300 words unless a detailed report is requested."""


def build_match_context(match_summary: dict) -> str:
    """Форматирует данные матча в текстовый контекст для LLM."""
    lines = ["=== MATCH DATA ==="]

    # Общая статистика
    if "teams" in match_summary:
        lines.append(f"\nTeams: {match_summary['teams']}")
    if "score" in match_summary:
        lines.append(f"Score: {match_summary['score']}")
    if "duration_min" in match_summary:
        lines.append(f"Duration: {match_summary['duration_min']} min")

    # Физические данные
    if "physical" in match_summary:
        phys = match_summary["physical"]
        lines.append("\n[PHYSICAL]")
        for team, stats in phys.items():
            lines.append(f"  {team}: dist={stats.get('distance_km', 0):.1f}km "
                         f"| sprints={stats.get('sprint_count', 0)} "
                         f"| max_speed={stats.get('max_speed', 0):.1f}km/h")

    # Тактика
    if "tactical" in match_summary:
        tac = match_summary["tactical"]
        lines.append("\n[TACTICAL]")
        if "formation" in tac:
            lines.append(f"  Formation: {tac['formation']}")
        if "pressing_style" in tac:
            lines.append(f"  Pressing: {tac['pressing_style']} "
                         f"(intensity={tac.get('pressing_intensity', 0):.2f})")
        if "territory_pct" in tac:
            lines.append(f"  Territory: {tac['territory_pct']:.0f}% in opponent half")
        if "compactness" in tac:
            lines.append(f"  Compactness: {tac['compactness']:.2f}")

    # Ошибки
    if "mistakes" in match_summary:
        mistakes = match_summary["mistakes"]
        lines.append("\n[MISTAKES]")
        for m in mistakes[:5]:  # top 5
            lines.append(f"  [{m.get('severity','?').upper()}] {m.get('type','?')} "
                         f"@ frame {m.get('frame_idx', 0)} "
                         f"(player {m.get('player_id', '?')})")

    # Рейтинги игроков
    if "player_ratings" in match_summary:
        ratings = match_summary["player_ratings"]
        lines.append("\n[PLAYER RATINGS] (top 5)")
        sorted_r = sorted(ratings, key=lambda x: x.get("overall", 0), reverse=True)
        for r in sorted_r[:5]:
            lines.append(f"  Player {r.get('player_id','?')}: "
                         f"overall={r.get('overall', 0):.1f} "
                         f"| physical={r.get('physical', 0):.1f} "
                         f"| tactical={r.get('tactical', 0):.1f}")

    # Усталость
    if "fatigue" in match_summary:
        fat = match_summary["fatigue"]
        lines.append("\n[FATIGUE]")
        if fat.get("critical_players"):
            lines.append(f"  CRITICAL: players {fat['critical_players']}")
        if fat.get("high_risk_players"):
            lines.append(f"  HIGH RISK: players {fat['high_risk_players']}")
        lines.append(f"  Team avg fatigue: {fat.get('team_fatigue_avg', 0):.1f}/100")

    # Предсказание
    if "prediction" in match_summary:
        pred = match_summary["prediction"]
        lines.append("\n[PREDICTION]")
        lines.append(f"  Outcome: {pred.get('outcome', '?')} "
                     f"(confidence={pred.get('confidence', 0):.0%})")

    lines.append("\n=== END OF DATA ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _AnthropicBackend:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def ask(self, question: str, context: str) -> CoachResponse:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
            ],
        )
        answer = msg.content[0].text
        tokens = msg.usage.input_tokens + msg.usage.output_tokens
        return CoachResponse(answer=answer, backend="anthropic",
                             tokens_used=tokens, model=self.model)


class _OpenAIBackend:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def ask(self, question: str, context: str) -> CoachResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
            ],
            max_tokens=1024,
        )
        answer = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return CoachResponse(answer=answer, backend="openai",
                             tokens_used=tokens, model=self.model)


class _OllamaBackend:
    """
    Бесплатный локальный LLM через Ollama (https://ollama.com).

    Установка:
        1. Скачать Ollama: https://ollama.com/download
        2. ollama pull llama3        (или mistral, gemma2, etc.)
        3. Запустить: ollama serve

    Переменные окружения:
        OLLAMA_BASE_URL  — default: http://localhost:11434
        OLLAMA_MODEL     — default: llama3
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def ask(self, question: str, context: str) -> CoachResponse:
        import urllib.request
        import json as _json

        payload = _json.dumps({
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT}\n\n{context}\n\nQuestion: {question}",
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = _json.loads(resp.read())

        answer = data.get("response", "").strip()
        return CoachResponse(
            answer=answer,
            backend="ollama",
            tokens_used=data.get("eval_count", 0),
            model=self.model,
        )


class _RuleBasedBackend:
    """Работает без API ключей — правила на основе данных."""

    def ask(self, question: str, context: str) -> CoachResponse:
        q = question.lower()
        answer = self._route(q, context)
        return CoachResponse(answer=answer, backend="rule_based", model="rule_based")

    def _route(self, q: str, context: str) -> str:
        if any(w in q for w in ["press", "прессинг", "давлен"]):
            return self._pressing_analysis(context)
        elif any(w in q for w in ["замен", "substitut", "устал", "fatig"]):
            return self._substitution_advice(context)
        elif any(w in q for w in ["ошибк", "mistake", "defend", "защит"]):
            return self._mistake_analysis(context)
        elif any(w in q for w in ["рейтинг", "rating", "лучш", "best"]):
            return self._rating_summary(context)
        elif any(w in q for w in ["тактик", "tactic", "схем", "formati"]):
            return self._tactical_summary(context)
        else:
            return self._general_summary(context)

    def _extract_section(self, context: str, section: str) -> str:
        lines = context.split("\n")
        in_section = False
        result = []
        for line in lines:
            if f"[{section}]" in line:
                in_section = True
            elif line.startswith("[") and in_section:
                break
            elif in_section:
                result.append(line)
        return "\n".join(result).strip()

    def _pressing_analysis(self, context: str) -> str:
        tac = self._extract_section(context, "TACTICAL")
        return (
            f"Pressing analysis based on match data:\n{tac}\n\n"
            "Recommendation: If pressing intensity is below 0.3, the team is defending deep. "
            "Consider increasing the pressing line to win possession higher up the pitch. "
            "Monitor compactness to avoid gaps between lines."
        )

    def _substitution_advice(self, context: str) -> str:
        fat = self._extract_section(context, "FATIGUE")
        return (
            f"Fatigue assessment:\n{fat}\n\n"
            "Players marked CRITICAL should be substituted immediately to prevent injury. "
            "HIGH RISK players should be replaced within 10-15 minutes. "
            "Prioritize substitutions based on position importance and match situation."
        )

    def _mistake_analysis(self, context: str) -> str:
        mis = self._extract_section(context, "MISTAKES")
        return (
            f"Defensive mistakes detected:\n{mis}\n\n"
            "Focus on: (1) Reducing unmarked opponents in dangerous zones, "
            "(2) Maintaining formation shape, (3) Closing gaps between defenders. "
            "Review these situations in video analysis with the team."
        )

    def _rating_summary(self, context: str) -> str:
        rat = self._extract_section(context, "PLAYER RATINGS")
        return f"Top performers this match:\n{rat}\n\nFocus training on lower-rated players to improve team balance."

    def _tactical_summary(self, context: str) -> str:
        tac = self._extract_section(context, "TACTICAL")
        return f"Tactical overview:\n{tac}"

    def _general_summary(self, context: str) -> str:
        phys = self._extract_section(context, "PHYSICAL")
        tac = self._extract_section(context, "TACTICAL")
        fat = self._extract_section(context, "FATIGUE")
        return (
            f"Match summary:\n\n[Physical]\n{phys}\n\n"
            f"[Tactical]\n{tac}\n\n"
            f"[Fitness]\n{fat}"
        )


# ---------------------------------------------------------------------------
# CoachAssistant
# ---------------------------------------------------------------------------

class CoachAssistant:
    """
    LLM-ассистент тренера.

    Автоматически выбирает лучший доступный бэкенд:
    Anthropic → OpenAI → Rule-based.

    Args:
        anthropic_api_key:  Ключ Anthropic (или ANTHROPIC_API_KEY из env)
        openai_api_key:     Ключ OpenAI (или OPENAI_API_KEY из env)
        model:              Модель Claude (default: claude-haiku-4-5-20251001)
        ollama_model:       Модель Ollama (default: llama3)
        force_backend:      "anthropic" / "openai" / "ollama" / "rule_based"
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        ollama_model: str | None = None,
        force_backend: str | None = None,
    ) -> None:
        self._backend = self._init_backend(
            anthropic_api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
            ollama_model=ollama_model or os.getenv("OLLAMA_MODEL", "llama3"),
            force=force_backend,
        )
        logger.info("CoachAssistant initialized | backend={}", self._backend.__class__.__name__)

    def ask(
        self,
        question: str,
        match_summary: dict[str, Any] | None = None,
        extra_context: str = "",
    ) -> CoachResponse:
        """
        Задать вопрос ассистенту.

        Args:
            question:       Вопрос тренера на любом языке
            match_summary:  Данные матча (dict)
            extra_context:  Дополнительный текстовый контекст

        Returns:
            CoachResponse с ответом
        """
        context_parts = []
        if match_summary:
            context_parts.append(build_match_context(match_summary))
        if extra_context:
            context_parts.append(extra_context)
        context = "\n\n".join(context_parts)

        try:
            response = self._backend.ask(question=question, context=context)
            logger.info(
                "CoachAssistant | backend={} | tokens={} | q_len={}",
                response.backend, response.tokens_used, len(question),
            )
            return response
        except Exception as e:
            logger.error("CoachAssistant backend error: {} — falling back to rule_based", e)
            fallback = _RuleBasedBackend()
            return fallback.ask(question=question, context=context)

    def tactical_report(self, match_summary: dict[str, Any]) -> CoachResponse:
        """Генерировать полный тактический отчёт по матчу."""
        return self.ask(
            question=(
                "Generate a comprehensive tactical match report. "
                "Cover: (1) Overall performance, (2) Pressing effectiveness, "
                "(3) Key mistakes, (4) Top performers, (5) Recommendations for next match."
            ),
            match_summary=match_summary,
        )

    def substitution_advice(self, match_summary: dict[str, Any]) -> CoachResponse:
        """Рекомендации по заменам на основе усталости и данных матча."""
        return self.ask(
            question=(
                "Based on fatigue data and performance ratings, "
                "which players should be substituted and why? "
                "List in order of priority."
            ),
            match_summary=match_summary,
        )

    @property
    def backend_name(self) -> str:
        return self._backend.__class__.__name__

    # ------------------------------------------------------------------

    @staticmethod
    def _init_backend(
        anthropic_api_key: str | None,
        openai_api_key: str | None,
        model: str,
        ollama_model: str,
        force: str | None,
    ):
        if force == "rule_based":
            return _RuleBasedBackend()
        if force == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return _OllamaBackend(model=ollama_model, base_url=base_url)
        if force == "openai":
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY required")
            return _OpenAIBackend(openai_api_key)

        # Приоритет: Anthropic → OpenAI → Ollama → rule_based
        if anthropic_api_key:
            try:
                import anthropic  # noqa: F401
                return _AnthropicBackend(anthropic_api_key, model=model)
            except ImportError:
                logger.warning("anthropic package not installed — pip install anthropic")

        if openai_api_key:
            try:
                import openai  # noqa: F401
                return _OpenAIBackend(openai_api_key)
            except ImportError:
                logger.warning("openai package not installed — pip install openai")

        # Проверяем доступность Ollama
        try:
            import urllib.request
            urllib.request.urlopen(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
                timeout=2,
            )
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            logger.info("Ollama detected at {} — using {} model", base_url, ollama_model)
            return _OllamaBackend(model=ollama_model, base_url=base_url)
        except Exception:
            pass

        logger.warning(
            "No API keys or Ollama found — using rule_based backend (free, offline). "
            "For LLM quality: set ANTHROPIC_API_KEY, OPENAI_API_KEY, or run Ollama locally."
        )
        return _RuleBasedBackend()
