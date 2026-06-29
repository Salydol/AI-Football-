"""
FeatureExtractor — извлекает фичи для модели предсказания из трекинг/тактических данных.

Фичи делятся на:
  Physical  — дистанция, скорость, спринты, интенсивность обеих команд
  Tactical  — компактность, прессинг, формация, ошибки
  Momentum  — тренд последних N минут (растёт ли активность)
  Spatial   — позиция мяча, угроза воротам, территориальное преимущество

Все фичи нормализованы в [0, 1] для совместимости с любым классификатором.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque

import numpy as np

from fie.ratings.aggregator import MatchAggregator
from fie.tactical.pressing import PressingAnalyzer
from fie.tactical.compactness import CompactnessAnalyzer
from fie.tracking.pipeline import TrackingFrame


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

@dataclass
class MatchFeatures:
    """Полный вектор фич для предсказания исхода."""

    # --- Физика команды LEFT (нормализована) ---
    left_distance_norm: float = 0.0       # дистанция / benchmark
    left_speed_max_norm: float = 0.0      # макс скорость / 36 км/ч
    left_sprint_count_norm: float = 0.0   # спринтов / 40
    left_intensity_norm: float = 0.0      # высокоинтенсивных действий

    # --- Физика команды RIGHT ---
    right_distance_norm: float = 0.0
    right_speed_max_norm: float = 0.0
    right_sprint_count_norm: float = 0.0
    right_intensity_norm: float = 0.0

    # --- Тактика LEFT ---
    left_pressing_line: float = 0.0       # 0..1 позиция линии прессинга
    left_pressing_intensity: float = 0.0  # 0..1
    left_compactness: float = 0.0         # 0..1
    left_high_press_pct: float = 0.0      # % времени в high press

    # --- Тактика RIGHT ---
    right_pressing_line: float = 0.0
    right_pressing_intensity: float = 0.0
    right_compactness: float = 0.0
    right_high_press_pct: float = 0.0

    # --- Территориальное преимущество ---
    left_territory_pct: float = 0.5       # % времени мяч на половине LEFT
    ball_avg_x_norm: float = 0.5          # средняя позиция мяча (0..1)
    inter_team_distance_norm: float = 0.5 # дистанция между командами

    # --- Momentum (тренд последних 5 минут vs предыдущие 5 минут) ---
    left_momentum: float = 0.0            # >0.5 = растёт, <0.5 = падает
    right_momentum: float = 0.0

    # --- Игровое время ---
    match_time_pct: float = 0.0           # 0..1 (0=начало, 1=конец)

    def to_array(self) -> np.ndarray:
        """Конвертировать в numpy array для модели."""
        return np.array(list(asdict(self).values()), dtype=np.float32)

    def to_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}

    @property
    def feature_names(self) -> list[str]:
        return list(asdict(self).keys())


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Извлекает фичи из потока TrackingFrame.

    Поддерживает два режима:
      - incremental: вызывай update() на каждый кадр, get_features() в любой момент
      - batch: передай список кадров в extract_from_frames()
    """

    BENCH_DISTANCE = 11000.0   # м за 90 мин
    BENCH_SPEED = 36.0         # км/ч
    BENCH_SPRINTS = 40
    BENCH_INTENSITY = 80       # high-accel actions

    MOMENTUM_WINDOW = 150      # кадров (~6 сек при 25fps × 60сек = 1500 для 1 мин)

    def __init__(
        self,
        field_length: float = 105.0,
        field_width: float = 68.0,
        fps: float = 25.0,
        match_duration: float = 5400.0,
    ) -> None:
        self.field_length = field_length
        self.field_width = field_width
        self.fps = fps
        self.match_duration = match_duration

        self._agg_left = MatchAggregator(fps=fps)
        self._agg_right = MatchAggregator(fps=fps)
        self._pressing = PressingAnalyzer(field_length=field_length)
        self._compactness = CompactnessAnalyzer(field_length=field_length, field_width=field_width)

        self._frame_count = 0
        self._ball_x_history: deque[float] = deque(maxlen=self.MOMENTUM_WINDOW * 2)

        # Momentum: активность команды за последние N кадров
        self._left_speed_window: deque[float] = deque(maxlen=self.MOMENTUM_WINDOW)
        self._right_speed_window: deque[float] = deque(maxlen=self.MOMENTUM_WINDOW)
        self._left_speed_prev: deque[float] = deque(maxlen=self.MOMENTUM_WINDOW)
        self._right_speed_prev: deque[float] = deque(maxlen=self.MOMENTUM_WINDOW)

        # Pressing stats accumulated
        self._left_high_press_frames = 0
        self._right_high_press_frames = 0
        self._pressing_frames = 0
        self._left_pressing_line_sum = 0.0
        self._right_pressing_line_sum = 0.0
        self._left_pressing_intensity_sum = 0.0
        self._right_pressing_intensity_sum = 0.0

        # Compactness accumulated
        self._left_compact_sum = 0.0
        self._right_compact_sum = 0.0
        self._inter_dist_sum = 0.0
        self._compact_frames = 0

        # Territory
        self._ball_in_left_half = 0
        self._ball_frames = 0

    def update(self, frame: TrackingFrame) -> None:
        """Обновить состояние одним кадром."""
        self._frame_count += 1
        mid = self.field_length / 2.0

        players = frame.players
        ball = frame.ball

        left_players = [p for p in players if p.x < mid]
        right_players = [p for p in players if p.x >= mid]

        # Фиктивные TrackingFrame для агрегаторов (только для своей команды)
        from fie.tracking.pipeline import TrackingFrame as TF
        left_frame = TF(frame.frame_idx, frame.timestamp, left_players, ball)
        right_frame = TF(frame.frame_idx, frame.timestamp, right_players, ball)

        self._agg_left.update(left_frame)
        self._agg_right.update(right_frame)

        # Pressing
        left_snap, right_snap = self._pressing.analyze_frame(players, ball)
        if left_snap:
            self._pressing_frames += 1
            self._left_pressing_line_sum += left_snap.pressing_line_pct
            self._left_pressing_intensity_sum += left_snap.pressing_intensity
            if left_snap.high_press:
                self._left_high_press_frames += 1
        if right_snap:
            self._right_pressing_line_sum += right_snap.pressing_line_pct
            self._right_pressing_intensity_sum += right_snap.pressing_intensity
            if right_snap.high_press:
                self._right_high_press_frames += 1

        # Compactness
        compact = self._compactness.analyze(players)
        if compact.left:
            self._left_compact_sum += compact.left.compactness
            self._compact_frames += 1
        if compact.right:
            self._right_compact_sum += compact.right.compactness
        if compact.inter_team_distance is not None:
            self._inter_dist_sum += compact.inter_team_distance

        # Territory
        if ball:
            self._ball_frames += 1
            self._ball_x_history.append(ball.x)
            if ball.x < mid:
                self._ball_in_left_half += 1

        # Momentum — средняя скорость команды
        if left_players:
            self._left_speed_window.append(np.mean([p.speed for p in left_players]))
        if right_players:
            self._right_speed_window.append(np.mean([p.speed for p in right_players]))

    def get_features(self) -> MatchFeatures:
        """Вычислить вектор фич из накопленных данных."""
        f = MatchFeatures()

        n = max(self._frame_count, 1)
        pf = max(self._pressing_frames, 1)
        cf = max(self._compact_frames, 1)

        # --- Физика ---
        left_stats = self._agg_left.get_all_stats()
        right_stats = self._agg_right.get_all_stats()

        if left_stats:
            left_phys = [p for p, _ in left_stats.values()]
            f.left_distance_norm = min(
                sum(p.distance_total for p in left_phys) / max(self.BENCH_DISTANCE * len(left_phys), 1), 1.5
            )
            f.left_speed_max_norm = min(
                max(p.speed_max for p in left_phys) / self.BENCH_SPEED, 1.2
            )
            f.left_sprint_count_norm = min(
                sum(p.sprint_count for p in left_phys) / max(self.BENCH_SPRINTS * len(left_phys), 1), 1.5
            )
            f.left_intensity_norm = min(
                sum(p.accel_high_count for p in left_phys) / max(self.BENCH_INTENSITY * len(left_phys), 1), 1.5
            )

        if right_stats:
            right_phys = [p for p, _ in right_stats.values()]
            f.right_distance_norm = min(
                sum(p.distance_total for p in right_phys) / max(self.BENCH_DISTANCE * len(right_phys), 1), 1.5
            )
            f.right_speed_max_norm = min(
                max(p.speed_max for p in right_phys) / self.BENCH_SPEED, 1.2
            )
            f.right_sprint_count_norm = min(
                sum(p.sprint_count for p in right_phys) / max(self.BENCH_SPRINTS * len(right_phys), 1), 1.5
            )
            f.right_intensity_norm = min(
                sum(p.accel_high_count for p in right_phys) / max(self.BENCH_INTENSITY * len(right_phys), 1), 1.5
            )

        # --- Тактика ---
        f.left_pressing_line = self._left_pressing_line_sum / pf
        f.left_pressing_intensity = self._left_pressing_intensity_sum / pf
        f.left_compactness = self._left_compact_sum / cf
        f.left_high_press_pct = self._left_high_press_frames / pf

        f.right_pressing_line = self._right_pressing_line_sum / pf
        f.right_pressing_intensity = self._right_pressing_intensity_sum / pf
        f.right_compactness = self._right_compact_sum / cf
        f.right_high_press_pct = self._right_high_press_frames / pf

        # --- Территория ---
        if self._ball_frames > 0:
            f.left_territory_pct = self._ball_in_left_half / self._ball_frames
            f.ball_avg_x_norm = (
                float(np.mean(list(self._ball_x_history))) / self.field_length
                if self._ball_x_history else 0.5
            )
        f.inter_team_distance_norm = min(
            self._inter_dist_sum / cf / max(self.field_length, 1), 1.0
        )

        # --- Momentum ---
        if len(self._left_speed_window) >= 10:
            half = len(self._left_speed_window) // 2
            recent = np.mean(list(self._left_speed_window)[half:])
            older = np.mean(list(self._left_speed_window)[:half])
            f.left_momentum = float(np.clip(recent / max(older, 0.1) / 2, 0, 1))
        else:
            f.left_momentum = 0.5

        if len(self._right_speed_window) >= 10:
            half = len(self._right_speed_window) // 2
            recent = np.mean(list(self._right_speed_window)[half:])
            older = np.mean(list(self._right_speed_window)[:half])
            f.right_momentum = float(np.clip(recent / max(older, 0.1) / 2, 0, 1))
        else:
            f.right_momentum = 0.5

        # --- Время ---
        f.match_time_pct = min(self._frame_count / (self.fps * self.match_duration), 1.0)

        return f

    def extract_from_frames(self, frames: list[TrackingFrame]) -> MatchFeatures:
        """Batch-режим: обработать список кадров и вернуть фичи."""
        for frame in frames:
            self.update(frame)
        return self.get_features()
