"""
PressingAnalyzer — анализ прессинга и давления команды.

Метрики:
  - pressing_line_x    : X-координата линии прессинга (где команда начинает давить)
  - pressing_intensity : % игроков активно прессингующих (в радиусе PRESS_RADIUS от мяча)
  - ppda               : Passes Per Defensive Action — чем меньше, тем агрессивнее прессинг
                         (приближение без статистики пасов: кол-во игроков в 1/3 соперника)
  - high_press         : True если линия прессинга выше центра поля
  - mid_block          : True если команда сидит в среднем блоке

Алгоритм:
  1. Определить команду (по средней X-позиции)
  2. Найти минимальный X игрока прессингующей команды = линия прессинга
  3. Посчитать игроков в радиусе PRESS_RADIUS от мяча → intensity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque

import numpy as np

from fie.tracking.pipeline import TrackedBall, TrackedPlayer


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

PRESS_RADIUS_M = 5.0      # радиус "прессинга" вокруг мяча (метры / пикселы)
HIGH_PRESS_THRESHOLD = 0.6 # линия прессинга > 60% длины поля = high press
MID_BLOCK_THRESHOLD = 0.4  # линия прессинга 30–60% = средний блок


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PressingSnapshot:
    """Мгновенный срез метрик прессинга."""
    pressing_line_x: float      # X-линия прессинга (в координатах поля)
    pressing_line_pct: float    # % от длины поля (0..1)
    pressing_intensity: float   # 0..1 — доля игроков вблизи мяча
    players_near_ball: int      # Кол-во прессингующих
    high_press: bool
    mid_block: bool
    deep_block: bool
    team: str                   # "left" / "right"

    def to_dict(self) -> dict:
        return {
            "pressing_line_x": round(self.pressing_line_x, 1),
            "pressing_line_pct": round(self.pressing_line_pct, 3),
            "pressing_intensity": round(self.pressing_intensity, 3),
            "players_near_ball": self.players_near_ball,
            "high_press": self.high_press,
            "mid_block": self.mid_block,
            "deep_block": self.deep_block,
            "team": self.team,
        }


@dataclass
class PressingStats:
    """Накопленная статистика прессинга за матч/отрезок."""
    team: str
    avg_pressing_line_pct: float   # средняя линия прессинга
    avg_intensity: float           # средняя интенсивность
    high_press_time_pct: float     # % времени в high press
    mid_block_time_pct: float      # % времени в mid block
    deep_block_time_pct: float     # % времени в deep block
    total_frames: int

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "avg_pressing_line_pct": round(self.avg_pressing_line_pct, 3),
            "avg_intensity": round(self.avg_intensity, 3),
            "high_press_time_pct": round(self.high_press_time_pct, 3),
            "mid_block_time_pct": round(self.mid_block_time_pct, 3),
            "deep_block_time_pct": round(self.deep_block_time_pct, 3),
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class PressingAnalyzer:
    """
    Анализирует интенсивность и линию прессинга обеих команд.

    Args:
        field_length: Длина поля (105 для реального, пиксели для Identity)
        smoothing: Количество кадров для EMA-сглаживания
        press_radius: Радиус (в ед. поля) для определения прессинга
    """

    def __init__(
        self,
        field_length: float = 105.0,
        smoothing: int = 15,
        press_radius: float = PRESS_RADIUS_M,
    ) -> None:
        self.field_length = field_length
        self.smoothing = smoothing
        self.press_radius = press_radius

        # Накопленная статистика
        self._left_line_history: deque[float] = deque(maxlen=smoothing)
        self._right_line_history: deque[float] = deque(maxlen=smoothing)
        self._left_intensity_history: deque[float] = deque(maxlen=smoothing)
        self._right_intensity_history: deque[float] = deque(maxlen=smoothing)

        # Счётчики для stats
        self._left_frames = 0
        self._right_frames = 0
        self._left_high_press = 0
        self._left_mid_block = 0
        self._left_deep_block = 0
        self._right_high_press = 0
        self._right_mid_block = 0
        self._right_deep_block = 0
        self._left_line_sum = 0.0
        self._right_line_sum = 0.0
        self._left_intensity_sum = 0.0
        self._right_intensity_sum = 0.0

    def analyze_frame(
        self,
        players: list[TrackedPlayer],
        ball: TrackedBall | None,
    ) -> tuple[PressingSnapshot | None, PressingSnapshot | None]:
        """
        Анализировать один кадр.

        Returns:
            (left_team_snapshot, right_team_snapshot)
        """
        mid = self.field_length / 2.0

        left_players = [p for p in players if p.x < mid]
        right_players = [p for p in players if p.x >= mid]

        left_snap = self._analyze_team(left_players, ball, "left") if len(left_players) >= 4 else None
        right_snap = self._analyze_team(right_players, ball, "right") if len(right_players) >= 4 else None

        # Накопить статистику
        if left_snap:
            self._left_frames += 1
            self._left_line_sum += left_snap.pressing_line_pct
            self._left_intensity_sum += left_snap.pressing_intensity
            if left_snap.high_press: self._left_high_press += 1
            if left_snap.mid_block: self._left_mid_block += 1
            if left_snap.deep_block: self._left_deep_block += 1

        if right_snap:
            self._right_frames += 1
            self._right_line_sum += right_snap.pressing_line_pct
            self._right_intensity_sum += right_snap.pressing_intensity
            if right_snap.high_press: self._right_high_press += 1
            if right_snap.mid_block: self._right_mid_block += 1
            if right_snap.deep_block: self._right_deep_block += 1

        return left_snap, right_snap

    def get_stats(self) -> tuple[PressingStats | None, PressingStats | None]:
        """Вернуть накопленную статистику за весь период."""
        left_stats = None
        right_stats = None

        if self._left_frames > 0:
            n = self._left_frames
            left_stats = PressingStats(
                team="left",
                avg_pressing_line_pct=self._left_line_sum / n,
                avg_intensity=self._left_intensity_sum / n,
                high_press_time_pct=self._left_high_press / n,
                mid_block_time_pct=self._left_mid_block / n,
                deep_block_time_pct=self._left_deep_block / n,
                total_frames=n,
            )

        if self._right_frames > 0:
            n = self._right_frames
            right_stats = PressingStats(
                team="right",
                avg_pressing_line_pct=self._right_line_sum / n,
                avg_intensity=self._right_intensity_sum / n,
                high_press_time_pct=self._right_high_press / n,
                mid_block_time_pct=self._right_mid_block / n,
                deep_block_time_pct=self._right_deep_block / n,
                total_frames=n,
            )

        return left_stats, right_stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze_team(
        self,
        team_players: list[TrackedPlayer],
        ball: TrackedBall | None,
        team: str,
    ) -> PressingSnapshot:
        xs = [p.x for p in team_players]

        # Линия прессинга = самый выдвинутый игрок вперёд
        if team == "left":
            pressing_line_x = max(xs)   # левая команда атакует вправо
        else:
            pressing_line_x = min(xs)   # правая команда атакует влево

        pressing_line_pct = pressing_line_x / max(self.field_length, 1.0)

        # Интенсивность прессинга — игроки вблизи мяча
        players_near = 0
        if ball is not None:
            for p in team_players:
                dist = math.hypot(p.x - ball.x, p.y - ball.y)
                if dist <= self.press_radius:
                    players_near += 1

        intensity = players_near / max(len(team_players), 1)

        # Классификация блока
        if team == "left":
            line_pct = pressing_line_pct
        else:
            line_pct = 1.0 - pressing_line_pct

        high_press = line_pct >= HIGH_PRESS_THRESHOLD
        mid_block = MID_BLOCK_THRESHOLD <= line_pct < HIGH_PRESS_THRESHOLD
        deep_block = line_pct < MID_BLOCK_THRESHOLD

        return PressingSnapshot(
            pressing_line_x=pressing_line_x,
            pressing_line_pct=pressing_line_pct,
            pressing_intensity=intensity,
            players_near_ball=players_near,
            high_press=high_press,
            mid_block=mid_block,
            deep_block=deep_block,
            team=team,
        )
