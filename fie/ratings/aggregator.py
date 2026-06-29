"""
PlayerMetricsAggregator — собирает статистику игрока за матч из трекинг-данных.

Принимает поток TrackingFrame и накапливает:
  - дистанцию (total, walking, jogging, running, sprinting)
  - скорости (max, avg)
  - ускорения (max, count высоких ускорений)
  - количество спринтов
  - тепловую карту позиций (для tactical анализа)
  - время владения мячом (proximity к мячу)

Использование:
    agg = MatchAggregator(fps=25)
    for frame in pipeline.process(source):
        agg.update(frame)
    stats = agg.get_all_stats()
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from fie.tracking.pipeline import TrackingFrame

# ---------------------------------------------------------------------------
# Speed zones (км/ч) — стандарт FIFA / Opta
# ---------------------------------------------------------------------------

ZONE_WALK = (0.0, 7.0)
ZONE_JOG = (7.0, 14.0)
ZONE_RUN = (14.0, 21.0)
ZONE_HIGH_RUN = (21.0, 25.0)
ZONE_SPRINT = (25.0, 100.0)

SPRINT_THRESHOLD = 25.0     # км/ч — начало спринта
HIGH_ACCEL_THRESHOLD = 2.5  # м/с² — высокое ускорение
BALL_PROXIMITY_M = 3.0      # метры — "владение" мячом


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PhysicalStats:
    """Физические метрики игрока за матч."""
    player_id: int

    # Дистанция (метры)
    distance_total: float = 0.0
    distance_walk: float = 0.0
    distance_jog: float = 0.0
    distance_run: float = 0.0
    distance_high_run: float = 0.0
    distance_sprint: float = 0.0

    # Скорость
    speed_max: float = 0.0
    speed_avg: float = 0.0
    speed_sum: float = 0.0
    speed_samples: int = 0

    # Ускорение
    accel_max: float = 0.0
    accel_high_count: int = 0   # количество ускорений > HIGH_ACCEL_THRESHOLD

    # Спринты
    sprint_count: int = 0
    sprint_distance: float = 0.0
    _in_sprint: bool = field(default=False, repr=False)

    # Время на поле (секунды)
    time_on_field: float = 0.0

    # Близость к мячу (секунды)
    time_near_ball: float = 0.0

    def update_speed_avg(self) -> None:
        if self.speed_samples > 0:
            self.speed_avg = self.speed_sum / self.speed_samples


@dataclass
class PositionalStats:
    """Позиционные метрики для tactical анализа."""
    player_id: int

    # Тепловая карта: сетка 10×7 на поле 105×68
    heatmap: np.ndarray = field(
        default_factory=lambda: np.zeros((10, 7), dtype=np.float32)
    )

    # Средняя позиция (centre of gravity)
    avg_x: float = 0.0
    avg_y: float = 0.0
    _pos_sum_x: float = 0.0
    _pos_sum_y: float = 0.0
    _pos_count: int = 0

    # Зоны активности
    defensive_third_time: float = 0.0   # x < 35
    middle_third_time: float = 0.0       # 35 ≤ x < 70
    attacking_third_time: float = 0.0    # x ≥ 70

    def add_position(self, x: float, y: float, dt: float) -> None:
        # Heatmap (clamp to pitch)
        gx = min(int(x / 105.0 * 10), 9)
        gy = min(int(y / 68.0 * 7), 6)
        self.heatmap[gx, gy] += dt

        # Running average position
        self._pos_sum_x += x
        self._pos_sum_y += y
        self._pos_count += 1
        if self._pos_count > 0:
            self.avg_x = self._pos_sum_x / self._pos_count
            self.avg_y = self._pos_sum_y / self._pos_count

        # Zone time
        if x < 35:
            self.defensive_third_time += dt
        elif x < 70:
            self.middle_third_time += dt
        else:
            self.attacking_third_time += dt


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class MatchAggregator:
    """
    Накапливает статистику всех игроков за матч.

    Вызывайте update() на каждый TrackingFrame из pipeline.
    В конце вызывайте get_all_stats() для получения результатов.
    """

    def __init__(self, fps: float = 25.0) -> None:
        self.fps = fps
        self._dt = 1.0 / fps  # секунды между кадрами

        self._physical: dict[int, PhysicalStats] = {}
        self._positional: dict[int, PositionalStats] = {}
        self._prev_positions: dict[int, tuple[float, float]] = {}
        self._frame_count = 0

    def update(self, frame: TrackingFrame) -> None:
        """Обработать один кадр."""
        self._frame_count += 1
        dt = self._dt

        for player in frame.players:
            pid = player.player_id

            # Инициализация при первом появлении
            if pid not in self._physical:
                self._physical[pid] = PhysicalStats(player_id=pid)
                self._positional[pid] = PositionalStats(player_id=pid)

            phys = self._physical[pid]
            pos = self._positional[pid]

            # --- Время на поле ---
            phys.time_on_field += dt

            # --- Скорость ---
            spd = player.speed  # км/ч
            phys.speed_max = max(phys.speed_max, spd)
            phys.speed_sum += spd
            phys.speed_samples += 1

            # --- Ускорение ---
            accel = abs(player.acceleration)
            phys.accel_max = max(phys.accel_max, accel)
            if accel > HIGH_ACCEL_THRESHOLD:
                phys.accel_high_count += 1

            # --- Дистанция по зонам ---
            if pid in self._prev_positions:
                px, py = self._prev_positions[pid]
                dist = math.hypot(player.x - px, player.y - py)
                phys.distance_total += dist

                if spd < ZONE_WALK[1]:
                    phys.distance_walk += dist
                elif spd < ZONE_JOG[1]:
                    phys.distance_jog += dist
                elif spd < ZONE_RUN[1]:
                    phys.distance_run += dist
                elif spd < ZONE_HIGH_RUN[1]:
                    phys.distance_high_run += dist
                else:
                    phys.distance_sprint += dist

            self._prev_positions[pid] = (player.x, player.y)

            # --- Спринты ---
            if spd >= SPRINT_THRESHOLD:
                if not phys._in_sprint:
                    phys.sprint_count += 1
                    phys._in_sprint = True
            else:
                phys._in_sprint = False

            # --- Близость к мячу ---
            if frame.ball is not None:
                dist_to_ball = math.hypot(
                    player.x - frame.ball.x,
                    player.y - frame.ball.y,
                )
                if dist_to_ball <= BALL_PROXIMITY_M:
                    phys.time_near_ball += dt

            # --- Позиционная статистика ---
            pos.add_position(player.x, player.y, dt)

        # Финализировать avg speed
        for phys in self._physical.values():
            phys.update_speed_avg()

    def get_physical(self, player_id: int) -> PhysicalStats | None:
        s = self._physical.get(player_id)
        if s:
            s.update_speed_avg()
        return s

    def get_positional(self, player_id: int) -> PositionalStats | None:
        return self._positional.get(player_id)

    def get_all_player_ids(self) -> list[int]:
        return sorted(self._physical.keys())

    def get_all_stats(self) -> dict[int, tuple[PhysicalStats, PositionalStats]]:
        """Вернуть все статистики: {player_id: (physical, positional)}"""
        result = {}
        for pid in self._physical:
            self._physical[pid].update_speed_avg()
            result[pid] = (self._physical[pid], self._positional[pid])
        return result

    @property
    def total_frames(self) -> int:
        return self._frame_count

    @property
    def match_duration_seconds(self) -> float:
        return self._frame_count * self._dt
