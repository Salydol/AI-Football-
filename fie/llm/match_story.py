"""
Match Story — AI нарратив матча.

Генерирует текстовый репортаж в стиле спортивного журналиста:
  - Предматчевый прогноз (если есть данные prediction)
  - Ключевые моменты по таймлайну
  - Анализ поворотных точек
  - Итоговый разбор

Использует CoachAssistant под капотом (Claude/OpenAI/rule_based).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fie.llm.coach_assistant import CoachAssistant, CoachResponse


# ---------------------------------------------------------------------------
# Story builder
# ---------------------------------------------------------------------------

STORY_SYSTEM_PROMPT = """You are a professional football journalist and analyst.
Write engaging, insightful match reports that combine data analysis with narrative storytelling.
Use specific numbers from the data. Be concise but vivid.
Respond in the same language as requested (default: English).
Structure: opening hook → key moments → tactical analysis → player highlights → conclusion."""


STORY_QUESTION = """Write a complete match story/report based on the data above.
Include:
1. A compelling opening that captures the match atmosphere
2. Key tactical battles and how they shaped the game
3. Standout individual performances (reference player IDs and ratings)
4. Critical mistakes and their impact
5. A conclusion with the main takeaway

Make it read like a high-quality sports report, not a dry statistics summary."""


STORY_QUESTION_RU = """Напиши полный репортаж о матче на основе данных выше.
Включи:
1. Захватывающее начало, передающее атмосферу матча
2. Ключевые тактические противостояния
3. Лучшие индивидуальные выступления (ссылайся на ID игроков и рейтинги)
4. Критические ошибки и их влияние на игру
5. Вывод с главным итогом матча

Пиши как профессиональный спортивный журналист, не просто сухую статистику."""


@dataclass
class MatchStory:
    text: str
    language: str
    backend: str
    tokens_used: int

    def to_dict(self) -> dict:
        return {
            "story": self.text,
            "language": self.language,
            "backend": self.backend,
            "tokens_used": self.tokens_used,
        }


# ---------------------------------------------------------------------------
# MatchStoryGenerator
# ---------------------------------------------------------------------------

class MatchStoryGenerator:
    """
    Генератор текстового нарратива матча.

    Args:
        assistant:  CoachAssistant (создаётся автоматически если не передан)
    """

    def __init__(self, assistant: CoachAssistant | None = None) -> None:
        self._assistant = assistant or CoachAssistant()

    def generate(
        self,
        match_summary: dict[str, Any],
        language: str = "en",
        home_team: str = "Home",
        away_team: str = "Away",
    ) -> MatchStory:
        """
        Сгенерировать нарратив матча.

        Args:
            match_summary:  Данные матча (то же что для CoachAssistant)
            language:       "en" или "ru"
            home_team:      Название домашней команды
            away_team:      Название гостевой команды

        Returns:
            MatchStory с текстом репортажа
        """
        # Добавляем названия команд в summary
        enriched = dict(match_summary)
        enriched.setdefault("teams", f"{home_team} vs {away_team}")

        question = STORY_QUESTION_RU if language == "ru" else STORY_QUESTION

        # Используем кастомный системный промпт для журналиста
        original_backend = self._assistant._backend

        # Подменяем промпт на журналистский
        class JournalistAdapter:
            def __init__(self, base):
                self._base = base

            def ask(self, question: str, context: str) -> CoachResponse:
                # Для rule_based — строим нарратив из правил
                if hasattr(self._base, "_route"):
                    return self._rule_based_story(context, language)
                # Для LLM бэкендов — используем специальный системный промпт
                if hasattr(self._base, "client"):
                    return self._llm_story(question, context)
                return self._base.ask(question, context)

            def _llm_story(self, question: str, context: str) -> CoachResponse:
                # Anthropic
                if hasattr(self._base, "client") and hasattr(self._base.client, "messages"):
                    msg = self._base.client.messages.create(
                        model=self._base.model,
                        max_tokens=2048,
                        system=STORY_SYSTEM_PROMPT,
                        messages=[
                            {"role": "user", "content": f"{context}\n\n{question}"}
                        ],
                    )
                    text = msg.content[0].text
                    tokens = msg.usage.input_tokens + msg.usage.output_tokens
                    return CoachResponse(text, "anthropic", tokens, self._base.model)
                return self._base.ask(question, context)

            def _rule_based_story(self, context: str, lang: str) -> CoachResponse:
                story = _build_rule_based_story(context, lang)
                return CoachResponse(story, "rule_based", 0, "rule_based")

        self._assistant._backend = JournalistAdapter(original_backend)
        try:
            response = self._assistant.ask(question=question, match_summary=enriched)
        finally:
            self._assistant._backend = original_backend

        return MatchStory(
            text=response.answer,
            language=language,
            backend=response.backend,
            tokens_used=response.tokens_used,
        )

    def key_moments_summary(
        self,
        events: list[dict],
        language: str = "en",
    ) -> str:
        """
        Короткое текстовое описание ключевых моментов из списка событий.

        Args:
            events: список ClipEvent.to_dict() или EventResult.to_dict()
            language: "en" / "ru"
        """
        if not events:
            return "No key moments detected." if language == "en" else "Ключевых моментов не обнаружено."

        lines = []
        for e in events:
            t = e.get("timestamp", 0)
            etype = e.get("event_type") or e.get("event", "event")
            conf = e.get("confidence", 1.0)
            pid = e.get("player_id")

            m = int(t // 60)
            s = int(t % 60)

            pid_str = f" (player #{pid})" if pid else ""
            if language == "ru":
                lines.append(f"  {m:02d}:{s:02d} — {_event_name_ru(etype)}{pid_str} (уверенность: {conf:.0%})")
            else:
                lines.append(f"  {m:02d}:{s:02d} — {etype.upper()}{pid_str} (conf: {conf:.0%})")

        header = "Ключевые моменты:" if language == "ru" else "Key moments:"
        return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule-based story builder
# ---------------------------------------------------------------------------

def _event_name_ru(etype: str) -> str:
    mapping = {
        "shot": "удар по воротам",
        "pass": "передача",
        "tackle": "отбор мяча",
        "dribble": "дриблинг",
        "clearance": "выбивание мяча",
        "ball_receipt": "приём мяча",
        "carry": "ведение мяча",
        "background": "фон",
    }
    return mapping.get(etype.lower(), etype)


def _build_rule_based_story(context: str, language: str) -> str:
    """Строит нарратив без LLM из структурированных данных."""

    def extract(section: str) -> str:
        lines = context.split("\n")
        in_sec = False
        result = []
        for line in lines:
            if f"[{section}]" in line:
                in_sec = True
            elif line.startswith("[") and in_sec:
                break
            elif in_sec and line.strip():
                result.append(line.strip())
        return "\n".join(result)

    phys = extract("PHYSICAL")
    tac = extract("TACTICAL")
    mistakes = extract("MISTAKES")
    ratings = extract("PLAYER RATINGS")
    fatigue = extract("FATIGUE")
    prediction = extract("PREDICTION")

    if language == "ru":
        story = f"""МАТЧЕВЫЙ ОТЧЁТ
{'='*50}

ФИЗИЧЕСКАЯ ПОДГОТОВКА
{phys or 'Данные недоступны.'}

ТАКТИЧЕСКИЙ АНАЛИЗ
{tac or 'Данные недоступны.'}

КЛЮЧЕВЫЕ ОШИБКИ
{mistakes or 'Критических ошибок не обнаружено.'}

ЛУЧШИЕ ИГРОКИ
{ratings or 'Данные о рейтингах недоступны.'}

СОСТОЯНИЕ ИГРОКОВ
{fatigue or 'Данные об усталости недоступны.'}

ПРОГНОЗ
{prediction or 'Прогноз недоступен.'}

{'='*50}
Отчёт сгенерирован Football Intelligence Engine (rule-based mode).
Подключите API ключ Anthropic для расширенного анализа: ANTHROPIC_API_KEY
"""
    else:
        story = f"""MATCH REPORT
{'='*50}

PHYSICAL PERFORMANCE
{phys or 'Data unavailable.'}

TACTICAL ANALYSIS
{tac or 'Data unavailable.'}

KEY MISTAKES
{mistakes or 'No critical mistakes detected.'}

TOP PERFORMERS
{ratings or 'Rating data unavailable.'}

PLAYER FITNESS
{fatigue or 'Fatigue data unavailable.'}

PREDICTION
{prediction or 'Prediction unavailable.'}

{'='*50}
Report generated by Football Intelligence Engine (rule-based mode).
Connect Anthropic API key for enhanced analysis: ANTHROPIC_API_KEY
"""
    return story
