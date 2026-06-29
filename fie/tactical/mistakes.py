"""
MistakeDetector — определяет тактические ошибки игроков.

Типы ошибок:
  1. UNMARKED_OPPONENT   — соперник оказался один без опеки в опасной зоне
  2. LINE_BREAK          — игрок выбился из своей линии (защита/полузащита)
  3. WRONG_ZONE          — игрок долго находится не в своей трети поля
  4. EXPOSED_SPACE       — большая дыра в обороне (зона без игроков)
  5. PRESSING_MISMATCH   — один игрок не участвует в командном прессинге
  6. OFFSIDE_RISK        — защитник слишком глубоко, риск офсайда

Алгоритм:
  - Каждый кадр проверяет несколько правил
  - Ошибка фиксируется только если длится > MIN_DURATION кадров (фильтр шума)
  - Каждая ошибка имеет severity: LOW / MEDIUM / HIGH
  - Ошибки одного типа для одного игрока группируются в инциденты

Использование:
    detector = MistakeDetector(field_length=105, field_width=68)
    for frame in tracking_frames:
        mistakes = detector.analyze(frame, formation_left, formation_right)
        for m in mistakes:
            print(m.to_dict())
    summary = detector.get_summary()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

import numpy as np

from fie.tactical.formation import FormationResult
from fie.tracking.pipeline import TrackedBall, TrackedPlayer, TrackingFrame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_DURATION_FRAMES = 8       # минимум кадров для фиксации ошибки
UNMARKED_RADIUS = 6.0         # метры — зона опеки вокруг соперника
DANGEROUS_ZONE_X = 0.65       # % от длины поля — опасная зона
LINE_BREAK_THRESHOLD = 8.0    # метры — отклонение от линии
EXPOSED_SPACE_MIN = 15.0      # м² — минимальная "дыра" в обороне
PRESSING_ISOLATION_RADIUS = 20.0  # метры — изоляция от команды при прессинге


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class MistakeType(str, Enum):
    UNMARKED_OPPONENT = "unmarked_opponent"
    LINE_BREAK = "line_break"
    WRONG_ZONE = "wrong_zone"
    EXPOSED_SPACE = "exposed_space"
    PRESSING_MISMATCH = "pressing_mismatch"
    OFFSIDE_RISK = "offside_risk"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Mistake:
    """Одна зафиксированная тактическая ошибка."""
    mistake_type: MistakeType
    severity: Severity
    player_id: int | None          # виновный игрок (None для командных ошибок)
    opponent_id: int | None        # соперник (для UNMARKED)
    frame_idx: int
    timestamp: float
    x: float                       # координата ошибки
    y: float
    description: str               # читаемое описание
    duration_frames: int = 1       # сколько кадров длилась ошибка

    def to_dict(self) -> dict:
        return {
            "type": self.mistake_type.value,
            "severity": self.severity.value,
            "player_id": self.player_id,
            "opponent_id": self.opponent_id,
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 2),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "description": self.description,
            "duration_frames": self.duration_frames,
        }


@dataclass
class MistakeSummary:
    """Сводка ошибок за матч."""
    total_mistakes: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    by_player: dict[int, int]       # player_id → кол-во ошибок
    worst_player_id: int | None
    most_common_type: str | None
    mistakes: list[Mistake]

    def to_dict(self) -> dict:
        return {
            "total_mistakes": self.total_mistakes,
            "by_type": self.by_type,
            "by_severity": self.by_severity,
            "by_player": {str(k): v for k, v in self.by_player.items()},
            "worst_player_id": self.worst_player_id,
            "most_common_type": self.most_common_type,
            "mistakes": [m.to_dict() for m in self.mistakes],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class MistakeDetector:
    """
    Детектор тактических ошибок.

    Args:
        field_length:     Длина поля (105 или пиксели)
        field_width:      Ширина поля (68 или пиксели)
        fps:              Кадров в секунду (для временных меток)
        min_duration:     Минимум кадров для фиксации ошибки
        unmarked_radius:  Радиус опеки в единицах поля
    """

    def __init__(
        self,
        field_length: float = 105.0,
        field_width: float = 68.0,
        fps: float = 25.0,
        min_duration: int = MIN_DURATION_FRAMES,
        unmarked_radius: float = UNMARKED_RADIUS,
    ) -> None:
        self.field_length = field_length
        self.field_width = field_width
        self.fps = fps
        self.min_duration = min_duration
        self.unmarked_radius = unmarked_radius

        # Активные "кандидаты" на ошибку: (player_id, type) → счётчик кадров
        self._candidates: dict[tuple, int] = defaultdict(int)
        self._candidate_data: dict[tuple, dict] = {}

        # Зафиксированные ошибки
        self._mistakes: list[Mistake] = []
        self._active_mistakes: set[tuple] = set()  # что сейчас активно

    def analyze(
        self,
        frame: TrackingFrame,
        formation_left: FormationResult | None = None,
        formation_right: FormationResult | None = None,
    ) -> list[Mistake]:
        """
        Анализировать один кадр.

        Returns:
            Список новых ошибок зафиксированных в этом кадре.
        """
        players = frame.players
        ball = frame.ball
        mid = self.field_length / 2.0

        left_players = [p for p in players if p.x < mid]
        right_players = [p for p in players if p.x >= mid]

        # Кандидаты в этом кадре
        current_candidates: set[tuple] = set()
        new_mistakes: list[Mistake] = []

        # --- Правило 1: Незакрытый соперник ---
        new_candidates = self._check_unmarked_opponents(
            left_players, right_players, ball, frame
        )
        current_candidates.update(new_candidates)

        # --- Правило 2: Выход из линии ---
        for team, team_players, formation in [
            ("left", left_players, formation_left),
            ("right", right_players, formation_right),
        ]:
            if formation:
                cands = self._check_line_breaks(team_players, formation, frame)
                current_candidates.update(cands)

        # --- Правило 3: Неправильная зона ---
        for team, team_players, formation in [
            ("left", left_players, formation_left),
            ("right", right_players, formation_right),
        ]:
            if formation:
                cands = self._check_wrong_zone(team_players, formation, frame, team)
                current_candidates.update(cands)

        # --- Правило 4: Дыры в обороне ---
        cands = self._check_exposed_space(left_players, right_players, ball, frame)
        current_candidates.update(cands)

        # --- Правило 5: Прессинг-дисбаланс ---
        if ball:
            cands = self._check_pressing_mismatch(left_players, right_players, ball, frame)
            current_candidates.update(cands)

        # Обновить счётчики
        confirmed = self._update_candidates(current_candidates, frame)
        new_mistakes.extend(confirmed)

        self._mistakes.extend(confirmed)
        return confirmed

    def get_summary(self) -> MistakeSummary:
        """Сводка всех зафиксированных ошибок."""
        by_type: dict[str, int] = defaultdict(int)
        by_severity: dict[str, int] = defaultdict(int)
        by_player: dict[int, int] = defaultdict(int)

        for m in self._mistakes:
            by_type[m.mistake_type.value] += 1
            by_severity[m.severity.value] += 1
            if m.player_id is not None:
                by_player[m.player_id] += 1

        worst_player = max(by_player, key=by_player.get) if by_player else None
        most_common = max(by_type, key=by_type.get) if by_type else None

        return MistakeSummary(
            total_mistakes=len(self._mistakes),
            by_type=dict(by_type),
            by_severity=dict(by_severity),
            by_player=dict(by_player),
            worst_player_id=worst_player,
            most_common_type=most_common,
            mistakes=self._mistakes,
        )

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _check_unmarked_opponents(
        self,
        left: list[TrackedPlayer],
        right: list[TrackedPlayer],
        ball: TrackedBall | None,
        frame: TrackingFrame,
    ) -> set[tuple]:
        """Соперник без опеки в опасной зоне."""
        candidates: set[tuple] = set()
        dangerous_x = self.field_length * DANGEROUS_ZONE_X

        def _check(attackers, defenders, attack_dir):
            for att in attackers:
                # Только в опасной зоне
                in_danger = (
                    att.x > dangerous_x if attack_dir == "right"
                    else att.x < self.field_length - dangerous_x
                )
                if not in_danger:
                    continue

                # Есть ли рядом защитник?
                marked = any(
                    math.hypot(att.x - d.x, att.y - d.y) < self.unmarked_radius
                    for d in defenders
                )
                if not marked:
                    key = (att.player_id, MistakeType.UNMARKED_OPPONENT)
                    self._candidate_data[key] = {
                        "player_id": None,  # командная ошибка
                        "opponent_id": att.player_id,
                        "x": att.x,
                        "y": att.y,
                        "description": f"Player {att.player_id} unmarked in dangerous zone",
                        "severity": Severity.HIGH,
                        "type": MistakeType.UNMARKED_OPPONENT,
                        "frame_idx": frame.frame_idx,
                        "timestamp": frame.timestamp,
                    }
                    candidates.add(key)

        _check(right, left, "right")
        _check(left, right, "left")
        return candidates

    def _check_line_breaks(
        self,
        team_players: list[TrackedPlayer],
        formation: FormationResult,
        frame: TrackingFrame,
    ) -> set[tuple]:
        """Игрок выбился из своей линии формации."""
        candidates: set[tuple] = set()

        for line in formation.formation_lines:
            if line.count < 2:
                continue
            line_avg_x = line.avg_x

            for pid in line.player_ids:
                player = next((p for p in team_players if p.player_id == pid), None)
                if not player:
                    continue

                deviation = abs(player.x - line_avg_x)
                if deviation > LINE_BREAK_THRESHOLD:
                    key = (pid, MistakeType.LINE_BREAK)
                    severity = Severity.HIGH if deviation > LINE_BREAK_THRESHOLD * 2 else Severity.MEDIUM
                    self._candidate_data[key] = {
                        "player_id": pid,
                        "opponent_id": None,
                        "x": player.x,
                        "y": player.y,
                        "description": (
                            f"Player {pid} broke defensive line "
                            f"({deviation:.1f}m from line)"
                        ),
                        "severity": severity,
                        "type": MistakeType.LINE_BREAK,
                        "frame_idx": frame.frame_idx,
                        "timestamp": frame.timestamp,
                    }
                    candidates.add(key)

        return candidates

    def _check_wrong_zone(
        self,
        team_players: list[TrackedPlayer],
        formation: FormationResult,
        frame: TrackingFrame,
        team: str,
    ) -> set[tuple]:
        """Игрок долго находится не в своей трети поля."""
        candidates: set[tuple] = set()
        third = self.field_length / 3.0

        # Построить ожидаемую зону по позиции в формации
        for i, line in enumerate(formation.formation_lines):
            # i=0 — оборона, i=last — атака
            is_defensive_line = (i == 0)
            is_attack_line = (i == len(formation.formation_lines) - 1)

            for pid in line.player_ids:
                player = next((p for p in team_players if p.player_id == pid), None)
                if not player:
                    continue

                # Нападающий в своей трети — подозрительно
                if is_attack_line and team == "left" and player.x < third:
                    key = (pid, MistakeType.WRONG_ZONE)
                    self._candidate_data[key] = {
                        "player_id": pid,
                        "opponent_id": None,
                        "x": player.x,
                        "y": player.y,
                        "description": f"Forward {pid} stuck in own defensive third",
                        "severity": Severity.LOW,
                        "type": MistakeType.WRONG_ZONE,
                        "frame_idx": frame.frame_idx,
                        "timestamp": frame.timestamp,
                    }
                    candidates.add(key)

                # Защитник в чужой трети без мяча — подозрительно
                if is_defensive_line and team == "left" and player.x > third * 2:
                    key = (pid, MistakeType.WRONG_ZONE)
                    self._candidate_data[key] = {
                        "player_id": pid,
                        "opponent_id": None,
                        "x": player.x,
                        "y": player.y,
                        "description": f"Defender {pid} too far forward, leaving space",
                        "severity": Severity.MEDIUM,
                        "type": MistakeType.WRONG_ZONE,
                        "frame_idx": frame.frame_idx,
                        "timestamp": frame.timestamp,
                    }
                    candidates.add(key)

        return candidates

    def _check_exposed_space(
        self,
        left: list[TrackedPlayer],
        right: list[TrackedPlayer],
        ball: TrackedBall | None,
        frame: TrackingFrame,
    ) -> set[tuple]:
        """Большая дыра в обороне между игроками."""
        candidates: set[tuple] = set()
        if not ball:
            return candidates

        # Проверяем команду без мяча
        ball_left = ball.x < self.field_length / 2
        defending = right if ball_left else left
        team_key = "right" if ball_left else "left"

        if len(defending) < 4:
            return candidates

        # Сортируем защитников по X
        sorted_def = sorted(defending, key=lambda p: p.x)

        # Ищем большие промежутки
        for i in range(len(sorted_def) - 1):
            gap = sorted_def[i + 1].x - sorted_def[i].x
            if gap > EXPOSED_SPACE_MIN:
                gap_x = (sorted_def[i].x + sorted_def[i + 1].x) / 2
                gap_y = (sorted_def[i].y + sorted_def[i + 1].y) / 2

                # Дыра опасна если мяч близко
                dist_to_ball = math.hypot(gap_x - ball.x, gap_y - ball.y)
                if dist_to_ball < self.field_length * 0.3:
                    key = (sorted_def[i].player_id, MistakeType.EXPOSED_SPACE)
                    self._candidate_data[key] = {
                        "player_id": sorted_def[i].player_id,
                        "opponent_id": None,
                        "x": gap_x,
                        "y": gap_y,
                        "description": (
                            f"Exposed space ({gap:.1f}m gap) between "
                            f"players {sorted_def[i].player_id} and {sorted_def[i+1].player_id}"
                        ),
                        "severity": Severity.HIGH if gap > EXPOSED_SPACE_MIN * 1.5 else Severity.MEDIUM,
                        "type": MistakeType.EXPOSED_SPACE,
                        "frame_idx": frame.frame_idx,
                        "timestamp": frame.timestamp,
                    }
                    candidates.add(key)

        return candidates

    def _check_pressing_mismatch(
        self,
        left: list[TrackedPlayer],
        right: list[TrackedPlayer],
        ball: TrackedBall,
        frame: TrackingFrame,
    ) -> set[tuple]:
        """Игрок изолирован от команды во время прессинга."""
        candidates: set[tuple] = set()

        for team_players in [left, right]:
            if len(team_players) < 5:
                continue

            # Найти игроков близко к мячу (прессингующих)
            pressers = [
                p for p in team_players
                if math.hypot(p.x - ball.x, p.y - ball.y) < PRESSING_ISOLATION_RADIUS * 0.5
            ]
            if len(pressers) < 2:
                continue

            # Центр прессинга
            press_cx = np.mean([p.x for p in pressers])
            press_cy = np.mean([p.y for p in pressers])

            # Игрок который должен прессинговать, но далеко
            for p in team_players:
                dist_to_press = math.hypot(p.x - press_cx, p.y - press_cy)
                dist_to_ball = math.hypot(p.x - ball.x, p.y - ball.y)

                if dist_to_press > PRESSING_ISOLATION_RADIUS and dist_to_ball < self.field_length * 0.4:
                    key = (p.player_id, MistakeType.PRESSING_MISMATCH)
                    self._candidate_data[key] = {
                        "player_id": p.player_id,
                        "opponent_id": None,
                        "x": p.x,
                        "y": p.y,
                        "description": (
                            f"Player {p.player_id} not joining team press "
                            f"({dist_to_press:.1f}m from pressing group)"
                        ),
                        "severity": Severity.LOW,
                        "type": MistakeType.PRESSING_MISMATCH,
                        "frame_idx": frame.frame_idx,
                        "timestamp": frame.timestamp,
                    }
                    candidates.add(key)

        return candidates

    # ------------------------------------------------------------------
    # Candidate tracking (debounce)
    # ------------------------------------------------------------------

    def _update_candidates(
        self,
        current: set[tuple],
        frame: TrackingFrame,
    ) -> list[Mistake]:
        """
        Обновить счётчики кандидатов.
        Фиксировать ошибку только если она длится > min_duration кадров.
        """
        confirmed: list[Mistake] = []

        # Увеличить счётчик активных
        for key in current:
            self._candidates[key] += 1

        # Сбросить неактивные
        for key in list(self._candidates.keys()):
            if key not in current:
                self._candidates[key] = 0

        # Зафиксировать новые ошибки
        for key, count in self._candidates.items():
            if count == self.min_duration and key not in self._active_mistakes:
                data = self._candidate_data.get(key)
                if not data:
                    continue

                mistake = Mistake(
                    mistake_type=data["type"],
                    severity=data["severity"],
                    player_id=data["player_id"],
                    opponent_id=data.get("opponent_id"),
                    frame_idx=data["frame_idx"],
                    timestamp=data["timestamp"],
                    x=data["x"],
                    y=data["y"],
                    description=data["description"],
                    duration_frames=count,
                )
                confirmed.append(mistake)
                self._active_mistakes.add(key)

        # Убрать из активных то что прекратилось
        for key in list(self._active_mistakes):
            if self._candidates.get(key, 0) == 0:
                self._active_mistakes.discard(key)

        return confirmed
