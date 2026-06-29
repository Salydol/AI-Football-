"""
Query Engine -- search video index by natural language.

Two modes:
1. Rule-based (no API key): keyword matching, time filters, player id.
2. LLM-based (Anthropic/OpenAI): flexible NL understanding.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from loguru import logger

from fie.search.video_index import IndexedEvent, VideoIndex


# ---------------------------------------------------------------------------
# Parsed query
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    event_types: list[str]
    half: int | None
    minute_from: int | None
    minute_to: int | None
    player_id: int | None
    min_confidence: float
    description: str

    def to_dict(self) -> dict:
        return {
            "event_types": self.event_types,
            "half": self.half,
            "minute_from": self.minute_from,
            "minute_to": self.minute_to,
            "player_id": self.player_id,
            "min_confidence": self.min_confidence,
            "description": self.description,
        }


@dataclass
class QueryResult:
    query: str
    parsed: ParsedQuery
    matched_events: list[IndexedEvent]
    backend: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "backend": self.backend,
            "matched_count": len(self.matched_events),
            "parsed_query": self.parsed.to_dict(),
            "events": [e.to_dict() for e in self.matched_events],
        }


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

EVENT_ALIASES: dict[str, list[str]] = {
    "shot":         ["shot", "shots", "strike", "удар", "удары", "гол", "попытк"],
    "pass":         ["pass", "passes", "передача", "передач", "пасов", "пасы"],
    "tackle":       ["tackle", "tackles", "duel", "отбор", "перехват", "единоборств"],
    "dribble":      ["dribble", "dribbling", "обводка", "дриблинг", "финт"],
    "clearance":    ["clearance", "выбивание", "выбив"],
    "ball_receipt": ["ball_receipt", "receive", "reception", "приём"],
    "carry":        ["carry", "ведение", "run with ball"],
}

DANGER_ALIASES = ["опасн", "danger", "chance", "момент", "атак", "attack"]
BEST_ALIASES   = ["лучш", "best", "топ", "top", "highlight"]


def _match_event_type(query_lower: str) -> list[str]:
    # High-level categories take priority
    if any(a in query_lower for a in DANGER_ALIASES):
        return ["shot", "tackle", "dribble"]
    if any(a in query_lower for a in BEST_ALIASES):
        return ["shot", "tackle", "dribble", "clearance"]
    matched = []
    for etype, aliases in EVENT_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            matched.append(etype)
    return matched


def _parse_half(query_lower: str) -> int | None:
    first_kw  = ["первый тайм", "первом тайм", "first half", "1st half", "1-й тайм", "1 тайм"]
    second_kw = ["второй тайм", "втором тайм", "второго тайм",
                 "second half", "2nd half", "2-й тайм", "2 тайм"]
    if any(w in query_lower for w in first_kw):
        return 1
    if any(w in query_lower for w in second_kw):
        return 2
    return None


def _parse_time_range(query_lower: str) -> tuple[int | None, int | None]:
    minute_from = minute_to = None

    m = re.search(r"(?:после|after|from)\s+(\d+)\s*(?:мин|min)", query_lower)
    if m:
        minute_from = int(m.group(1))

    m = re.search(r"(?:до|before|until)\s+(\d+)\s*(?:мин|min)?", query_lower)
    if m:
        minute_to = int(m.group(1))

    m = re.search(r"(?:последни[ехй]|last)\s+(\d+)\s*(?:мин|min)", query_lower)
    if m:
        minute_from = max(0, 90 - int(m.group(1)))

    m = re.search(r"(?:с|between)\s+(\d+).{0,10}(?:по|to|and)\s+(\d+)", query_lower)
    if m:
        minute_from = int(m.group(1))
        minute_to   = int(m.group(2))

    return minute_from, minute_to


def _parse_player(query_lower: str) -> int | None:
    m = re.search(r"(?:игрок|player|#|номер|number)\s*#?(\d+)", query_lower)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Rule-based parser
# ---------------------------------------------------------------------------

class RuleBasedParser:
    def parse(self, query: str) -> ParsedQuery:
        q = query.lower()
        event_types = _match_event_type(q)
        half        = _parse_half(q)
        mfrom, mto  = _parse_time_range(q)
        player_id   = _parse_player(q)

        parts = []
        if event_types:
            parts.append("events: " + ", ".join(event_types))
        else:
            parts.append("all events")
        if half:
            parts.append(f"half {half}")
        if mfrom is not None:
            parts.append(f"from minute {mfrom}")
        if mto is not None:
            parts.append(f"to minute {mto}")
        if player_id:
            parts.append(f"player #{player_id}")

        return ParsedQuery(
            event_types=event_types,
            half=half,
            minute_from=mfrom,
            minute_to=mto,
            player_id=player_id,
            min_confidence=0.3,
            description=" | ".join(parts),
        )


# ---------------------------------------------------------------------------
# LLM parser
# ---------------------------------------------------------------------------

LLM_PARSE_PROMPT = (
    "You are a football video search assistant.\n"
    "Parse the user query into a structured JSON filter.\n"
    "Available event types: shot, pass, tackle, dribble, clearance, ball_receipt, carry\n"
    "Return ONLY valid JSON:\n"
    '{"event_types":[],"half":null,"minute_from":null,"minute_to":null,'
    '"player_id":null,"min_confidence":0.3,"description":""}\n\n'
    "Query: "
)


class LLMParser:
    def __init__(self, api_key: str, backend: str = "anthropic") -> None:
        self.api_key = api_key
        self.backend = backend

    def parse(self, query: str) -> ParsedQuery:
        try:
            if self.backend == "anthropic":
                return self._parse_anthropic(query)
            return self._parse_openai(query)
        except Exception as e:
            logger.warning("LLM parse failed: {} -- falling back to rule_based", e)
            return RuleBasedParser().parse(query)

    def _parse_anthropic(self, query: str) -> ParsedQuery:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": LLM_PARSE_PROMPT + query}],
        )
        return self._from_json(msg.content[0].text.strip())

    def _parse_openai(self, query: str) -> ParsedQuery:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": LLM_PARSE_PROMPT + query}],
            max_tokens=256,
        )
        return self._from_json(resp.choices[0].message.content or "{}")

    @staticmethod
    def _from_json(raw: str) -> ParsedQuery:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        data = json.loads(raw)
        return ParsedQuery(
            event_types=data.get("event_types") or [],
            half=data.get("half"),
            minute_from=data.get("minute_from"),
            minute_to=data.get("minute_to"),
            player_id=data.get("player_id"),
            min_confidence=data.get("min_confidence", 0.3),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------

class QueryEngine:
    """
    Search engine over VideoIndex.

    Auto-selects LLM parser if API key available, otherwise rule-based.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        ak = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        ok = openai_api_key or os.getenv("OPENAI_API_KEY")

        if ak:
            try:
                import anthropic  # noqa
                self._parser  = LLMParser(ak, "anthropic")
                self._backend = "anthropic"
                logger.info("QueryEngine: Anthropic LLM parser")
                return
            except ImportError:
                pass
        if ok:
            try:
                import openai  # noqa
                self._parser  = LLMParser(ok, "openai")
                self._backend = "openai"
                logger.info("QueryEngine: OpenAI LLM parser")
                return
            except ImportError:
                pass

        self._parser  = RuleBasedParser()
        self._backend = "rule_based"
        logger.info("QueryEngine: rule-based parser")

    def search(
        self,
        query: str,
        index: VideoIndex,
        max_results: int = 20,
    ) -> QueryResult:
        """Find events by natural language query."""
        parsed = self._parser.parse(query)
        logger.info(
            "Query '{}' -> types={} half={} min={}-{} player={}",
            query, parsed.event_types, parsed.half,
            parsed.minute_from, parsed.minute_to, parsed.player_id,
        )
        matched = index.search(
            event_types=parsed.event_types or None,
            half=parsed.half,
            minute_from=parsed.minute_from,
            minute_to=parsed.minute_to,
            player_id=parsed.player_id,
            min_confidence=parsed.min_confidence,
        )
        matched_sorted = sorted(matched, key=lambda e: e.confidence, reverse=True)
        return QueryResult(
            query=query,
            parsed=parsed,
            matched_events=matched_sorted[:max_results],
            backend=self._backend,
        )

    def search_and_clip(
        self,
        query: str,
        index: VideoIndex,
        video_path: str,
        output_dir: str = "clips",
        pre_seconds: float = 4.0,
        post_seconds: float = 3.0,
        max_clips: int = 10,
    ) -> dict:
        """Find events and extract video clips."""
        from fie.clipping.clipper import ClipEvent, ClipExtractor

        result = self.search(query, index, max_results=max_clips)
        if not result.matched_events:
            return {"query": query, "matched": 0, "clips": [], "parsed": result.parsed.to_dict()}

        extractor = ClipExtractor(
            video_path=video_path,
            output_dir=output_dir,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
        )
        clip_events = [
            ClipEvent(
                frame_idx=e.frame_idx,
                timestamp=e.timestamp_s,
                event_type=e.event_type,
                confidence=e.confidence,
                player_id=e.player_id,
            )
            for e in result.matched_events
        ]
        clips = extractor.extract_batch(clip_events)
        return {
            "query": query,
            "backend": self._backend,
            "matched": len(result.matched_events),
            "clips_extracted": len(clips),
            "parsed": result.parsed.to_dict(),
            "clips": [c.to_dict() for c in clips],
        }
