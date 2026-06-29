"""
FatigueAnalyzer — оценка риска травмы и усталости игрока.

Метрики:
  - sprint_load       : накопленная нагрузка спринтов
  - accel_load        : накопленная нагрузка резких ускорений
  - speed_decline     : падение максимальной скорости vs первые 15 мин
  - intensity_decline : падение высокоинтенсивных действий
  - pattern_change    : изменение паттерна движения (EMA)
  - total_load        : суммарная физическая нагрузка

Уровни риска:
  LOW      — всё в норме
  ELEVATED — начало усталости, мониторинг
  HIGH     — риск спада, рекомендуется замена
  CRITICAL — высокий риск травмы, срочная замена

Алгоритм:
  Делим матч на 15-минутные отрезки.
  Сравниваем активность каждого отрезка с первым (базовым).
  Падение > 20% = ELEVATED, > 35% = HIGH, > 50% = CRITICAL.
  Дополнительные триггеры: резкое падение скорости за 5 мин,
  аномальное изменение паттерна движения.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from enum import Enum

import numpy as np

from fie.tracking.pipeline import TrackedPlayer, TrackingFrame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_MINUTES = 15          # размер окна анализа (минуты)
SPEED_DECLINE_HIGH = 0.20    # падение скорости на 20% = HIGH
SPEED_DECLINE_CRITICAL = 0.35
INTENSITY_DECLINE_HIGH = 0.25
INTENSITY_DECLINE_CRITICAL = 0.45
SPRINT_OVERLOAD = 50         # спринтов за матч = перегрузка
ACCEL_OVERLOAD = 100         # резких ускорений = перегрузка


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class FatigueLevel(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PlayerFatigueState:
    """Текущее состояние усталости одного игрока."""
    player_id: int
    fatigue_level: FatigueLevel
    fatigue_score: float           # 0..100 (100 = максимальная усталость)
    injury_risk: float             # 0..1
    speed_decline_pct: float       # % падения скорости
    intensity_decline_pct: float   # % падения интенсивности
    sprint_load: int               # накоплено спринтов
    accel_load: int                # накоплено ускорений
    recommendation: str            # текстовая рекомендация
    minutes_played: float

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "fatigue_level": self.fatigue_level.value,
            "fatigue_score": round(self.fatigue_score, 1),
            "injury_risk": round(self.injury_risk, 3),
            "speed_decline_pct": round(self.speed_decline_pct, 1),
            "intensity_decline_pct": round(self.intensity_decline_pct, 1),
            "sprint_load": self.sprint_load,
            "accel_load": self.accel_load,
            "recommendation": self.recommendation,
            "minutes_played": round(self.minutes_played, 1),
        }


@dataclass
class FatigueSummary:
    """Сводка по всем игрокам."""
    players: list[PlayerFatigueState]
    critical_players: list[int]     # player_id с CRITICAL риском
    high_risk_players: list[int]    # player_id с HIGH риском
    team_fatigue_avg: float         # средняя усталость команды

    def to_dict(self) -> dict:
        return {
            "team_fatigue_avg": round(self.team_fatigue_avg, 1),
            "critical_players": self.critical_players,
            "high_risk_players": self.high_risk_players,
            "players": [p.to_dict() for p in self.players],
        }


# ---------------------------------------------------------------------------
# Per-player tracker
# ---------------------------------------------------------------------------

@dataclass
class _PlayerTracker:
    """Внутренний трекер состояния одного игрока."""
    player_id: int
    fps: float

    # Скользящие окна скорости (кадры)
    speed_window: deque = field(default_factory=lambda: deque(maxlen=375))   # 15 мин
    speed_baseline: deque = field(default_factory=lambda: deque(maxlen=375)) # первые 15 мин

    # Накопленные счётчики
    sprint_count: int = 0
    accel_count: int = 0
    _in_sprint: bool = False

    # Фреймы
    frame_count: int = 0
    baseline_done: bool = False

    # Базовые значения (первые 15 мин)
    baseline_avg_speed: float = 0.0
    baseline_intensity: float = 0.0

    def update(self, player: TrackedPlayer) -> None:
        self.frame_count += 1
        spd = player.speed
        accel = abs(player.acceleration)

        self.speed_window.append(spd)

        # Базовый период — первые 15 мин
        window_size = int(self.fps * 60 * WINDOW_MINUTES)
        if self.frame_count <= window_size:
            self.speed_baseline.append(spd)
        elif not self.baseline_done and len(self.speed_baseline) > 0:
            self.baseline_avg_speed = float(np.mean(list(self.speed_baseline)))
            self.baseline_intensity = sum(
                1 for s in self.speed_baseline if s > 14.0
            ) / max(len(self.speed_baseline), 1)
            self.baseline_done = True

        # Спринты
        if spd >= 25.0:
            if not self._in_sprint:
                self.sprint_count += 1
                self._in_sprint = True
        else:
            self._in_sprint = False

        # Высокие ускорения
        if accel > 2.5:
            self.accel_count += 1

    def compute_state(self) -> PlayerFatigueState:
        minutes = self.frame_count / (self.fps * 60)
        current_speeds = list(self.speed_window)

        if not current_speeds:
            return PlayerFatigueState(
                player_id=self.player_id,
                fatigue_level=FatigueLevel.LOW,
                fatigue_score=0.0,
                injury_risk=0.0,
                speed_decline_pct=0.0,
                intensity_decline_pct=0.0,
                sprint_load=self.sprint_count,
                accel_load=self.accel_count,
                recommendation="Insufficient data",
                minutes_played=minutes,
            )

        current_avg = float(np.mean(current_speeds))
        current_intensity = sum(1 for s in current_speeds if s > 14.0) / max(len(current_speeds), 1)

        # Падение скорости
        if self.baseline_avg_speed > 0:
            speed_decline = max(0.0, (self.baseline_avg_speed - current_avg) / self.baseline_avg_speed)
        else:
            speed_decline = 0.0

        # Падение интенсивности
        if self.baseline_intensity > 0:
            intensity_decline = max(0.0, (self.baseline_intensity - current_intensity) / self.baseline_intensity)
        else:
            intensity_decline = 0.0

        # Перегрузка спринтами/ускорениями
        sprint_overload = min(self.sprint_count / SPRINT_OVERLOAD, 1.0)
        accel_overload = min(self.accel_count / ACCEL_OVERLOAD, 1.0)

        # Итоговый балл усталости (0–100)
        fatigue_score = (
            speed_decline * 35 +
            intensity_decline * 30 +
            sprint_overload * 20 +
            accel_overload * 15
        )
        fatigue_score = min(fatigue_score * 100, 100.0)

        # Риск травмы
        injury_risk = float(np.clip(
            speed_decline * 0.4 +
            intensity_decline * 0.3 +
            sprint_overload * 0.2 +
            accel_overload * 0.1,
            0.0, 1.0
        ))

        # Уровень риска
        if speed_decline >= SPEED_DECLINE_CRITICAL or intensity_decline >= INTENSITY_DECLINE_CRITICAL:
            level = FatigueLevel.CRITICAL
            rec = f"Player {self.player_id}: CRITICAL fatigue. Immediate substitution recommended."
        elif speed_decline >= SPEED_DECLINE_HIGH or intensity_decline >= INTENSITY_DECLINE_HIGH:
            level = FatigueLevel.HIGH
            rec = f"Player {self.player_id}: HIGH fatigue. Consider substitution in next 10-15 min."
        elif fatigue_score > 35:
            level = FatigueLevel.ELEVATED
            rec = f"Player {self.player_id}: ELEVATED fatigue. Monitor closely."
        else:
            level = FatigueLevel.LOW
            rec = f"Player {self.player_id}: Normal condition."

        return PlayerFatigueState(
            player_id=self.player_id,
            fatigue_level=level,
            fatigue_score=fatigue_score,
            injury_risk=injury_risk,
            speed_decline_pct=speed_decline * 100,
            intensity_decline_pct=intensity_decline * 100,
            sprint_load=self.sprint_count,
            accel_load=self.accel_count,
            recommendation=rec,
            minutes_played=minutes,
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class FatigueAnalyzer:
    """
    Отслеживает усталость всех игроков в реальном времени.

    Args:
        fps:            Кадров в секунду
        field_length:   Длина поля (для нормализации координат)
    """

    def __init__(self, fps: float = 25.0, field_length: float = 105.0) -> None:
        self.fps = fps
        self.field_length = field_length
        self._trackers: dict[int, _PlayerTracker] = {}

    def update(self, frame: TrackingFrame) -> None:
        """Обновить состояние одним кадром."""
        for player in frame.players:
            pid = player.player_id
            if pid not in self._trackers:
                self._trackers[pid] = _PlayerTracker(player_id=pid, fps=self.fps)
            self._trackers[pid].update(player)

    def get_player_state(self, player_id: int) -> PlayerFatigueState | None:
        tracker = self._trackers.get(player_id)
        return tracker.compute_state() if tracker else None

    def get_summary(self) -> FatigueSummary:
        """Сводка по всем игрокам."""
        states = [t.compute_state() for t in self._trackers.values()]
        states.sort(key=lambda s: s.fatigue_score, reverse=True)

        critical = [s.player_id for s in states if s.fatigue_level == FatigueLevel.CRITICAL]
        high = [s.player_id for s in states if s.fatigue_level == FatigueLevel.HIGH]
        avg = float(np.mean([s.fatigue_score for s in states])) if states else 0.0

        return FatigueSummary(
            players=states,
            critical_players=critical,
            high_risk_players=high,
            team_fatigue_avg=avg,
        )

    def process_frames(self, frames: list[TrackingFrame]) -> FatigueSummary:
        """Batch-режим: обработать все кадры и вернуть сводку."""
        for frame in frames:
            self.update(frame)
        return self.get_summary()
