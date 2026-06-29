"""
RatingCalculator — переводит сырые метрики в рейтинги 0-100.

Physical, Technical, Tactical оценки зависят от позиции игрока.
Защитнику важен pressing и positioning, нападающему — speed и shooting.

Веса по позициям:
    GK  — goalkeeper
    DEF — defender (CB, LB, RB)
    MID — midfielder (CM, CDM, CAM, LM, RM)
    FWD — forward (ST, CF, LW, RW)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fie.ratings.aggregator import PhysicalStats, PositionalStats


# ---------------------------------------------------------------------------
# Position enum
# ---------------------------------------------------------------------------

class Position(str, Enum):
    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Rating result
# ---------------------------------------------------------------------------

@dataclass
class PhysicalRating:
    speed: float        # максимальная скорость
    acceleration: float # взрывные ускорения
    endurance: float    # дистанция / выносливость
    intensity: float    # спринты + high-intensity runs
    overall: float


@dataclass
class TacticalRating:
    positioning: float  # среднее положение в правильной зоне
    pressing: float     # активность прессинга (близость к мячу)
    coverage: float     # ширина покрытия поля
    overall: float


@dataclass
class PlayerRating:
    player_id: int
    position: Position
    physical: PhysicalRating
    tactical: TacticalRating
    overall: float      # взвешенный итог

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "position": self.position.value,
            "physical": {
                "speed": round(self.physical.speed, 1),
                "acceleration": round(self.physical.acceleration, 1),
                "endurance": round(self.physical.endurance, 1),
                "intensity": round(self.physical.intensity, 1),
                "overall": round(self.physical.overall, 1),
            },
            "tactical": {
                "positioning": round(self.tactical.positioning, 1),
                "pressing": round(self.tactical.pressing, 1),
                "coverage": round(self.tactical.coverage, 1),
                "overall": round(self.tactical.overall, 1),
            },
            "overall": round(self.overall, 1),
        }


# ---------------------------------------------------------------------------
# Benchmark values (per 90 minutes, elite level)
# Elite = 100 points. Average pro ≈ 65-70.
# ---------------------------------------------------------------------------

# Физические бенчмарки
BENCH_MAX_SPEED_KMH = 36.0       # топ скорость элиты (Мбаппе ~38)
BENCH_TOTAL_DISTANCE_M = 11000.0  # 11 км за 90 мин
BENCH_SPRINT_COUNT = 40
BENCH_HIGH_ACCEL = 80             # ускорений > 2.5 м/с²

# Тактические бенчмарки
BENCH_PRESSING_TIME_S = 180.0    # секунд у мяча (прессинг)


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _norm(value: float, benchmark: float, cap: float = 120.0) -> float:
    """Нормализует value относительно benchmark в [0, 100], с мягким cap."""
    if benchmark <= 0:
        return 0.0
    # Scale to 90 min equivalent
    ratio = value / benchmark
    # Logistic-ish: score = 100 * ratio^0.6 to reward high values but diminish returns
    score = 100.0 * (ratio ** 0.6)
    return min(score, cap)


def _weighted(*pairs: tuple[float, float]) -> float:
    """Взвешенное среднее: _weighted((value, weight), ...)"""
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in pairs) / total_w


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class RatingCalculator:
    """
    Вычисляет рейтинги игроков из агрегированной статистики.

    Масштабирует метрики к 90-минутному эквиваленту чтобы
    корректно сравнивать игроков с разным временем на поле.
    """

    def __init__(self, match_duration_seconds: float = 5400.0) -> None:
        """
        Args:
            match_duration_seconds: Длительность матча. Используется для
                масштабирования метрик к 90-минутному эквиваленту.
        """
        self.match_duration = match_duration_seconds
        self._scale = 5400.0 / max(match_duration_seconds, 1.0)  # к 90 мин

    def calculate(
        self,
        phys: PhysicalStats,
        pos: PositionalStats,
        position: Position = Position.UNKNOWN,
    ) -> PlayerRating:
        """Рассчитать полный рейтинг для одного игрока."""
        time = max(phys.time_on_field, 1.0)
        s = self._scale  # масштаб к 90 мин

        # ----------------------------------------------------------------
        # PHYSICAL
        # ----------------------------------------------------------------
        speed_score = _norm(phys.speed_max, BENCH_MAX_SPEED_KMH)
        accel_score = _norm(phys.accel_high_count * s, BENCH_HIGH_ACCEL)
        endurance_score = _norm(phys.distance_total * s, BENCH_TOTAL_DISTANCE_M)
        intensity_score = _norm(phys.sprint_count * s, BENCH_SPRINT_COUNT)

        # Веса зависят от позиции
        if position == Position.GK:
            phys_overall = _weighted(
                (speed_score, 0.2),
                (accel_score, 0.2),
                (endurance_score, 0.4),
                (intensity_score, 0.2),
            )
        elif position == Position.DEF:
            phys_overall = _weighted(
                (speed_score, 0.3),
                (accel_score, 0.3),
                (endurance_score, 0.2),
                (intensity_score, 0.2),
            )
        elif position == Position.MID:
            phys_overall = _weighted(
                (speed_score, 0.2),
                (accel_score, 0.2),
                (endurance_score, 0.4),
                (intensity_score, 0.2),
            )
        else:  # FWD / UNKNOWN
            phys_overall = _weighted(
                (speed_score, 0.4),
                (accel_score, 0.3),
                (endurance_score, 0.1),
                (intensity_score, 0.2),
            )

        physical = PhysicalRating(
            speed=speed_score,
            acceleration=accel_score,
            endurance=endurance_score,
            intensity=intensity_score,
            overall=phys_overall,
        )

        # ----------------------------------------------------------------
        # TACTICAL
        # ----------------------------------------------------------------

        # Positioning — насколько игрок в "правильной" трети
        total_time = max(
            pos.defensive_third_time + pos.middle_third_time + pos.attacking_third_time,
            1.0,
        )
        if position == Position.GK:
            positioning_score = (pos.defensive_third_time / total_time) * 100.0
        elif position == Position.DEF:
            positioning_score = (
                (pos.defensive_third_time * 0.6 + pos.middle_third_time * 0.4) / total_time
            ) * 100.0
        elif position == Position.MID:
            positioning_score = (
                (pos.middle_third_time * 0.6
                 + pos.defensive_third_time * 0.2
                 + pos.attacking_third_time * 0.2) / total_time
            ) * 100.0
        else:  # FWD
            positioning_score = (
                (pos.attacking_third_time * 0.6 + pos.middle_third_time * 0.4) / total_time
            ) * 100.0

        # Pressing — время у мяча относительно игрового времени
        pressing_score = _norm(phys.time_near_ball * s, BENCH_PRESSING_TIME_S)

        # Coverage — разброс позиций (стандартное отклонение по y)
        # Широкий игрок → больше coverage
        if pos._pos_count > 1:
            coverage_raw = pos.heatmap.sum(axis=0)  # распределение по ширине
            nonzero = coverage_raw[coverage_raw > 0]
            if len(nonzero) >= 2:
                coverage_score = min(
                    (nonzero.std() / 3.5) * 100.0,  # 3.5 = половина ширины
                    100.0,
                )
            else:
                coverage_score = 20.0
        else:
            coverage_score = 0.0

        if position == Position.GK:
            tact_overall = _weighted(
                (positioning_score, 0.6),
                (pressing_score, 0.1),
                (coverage_score, 0.3),
            )
        elif position == Position.DEF:
            tact_overall = _weighted(
                (positioning_score, 0.4),
                (pressing_score, 0.4),
                (coverage_score, 0.2),
            )
        elif position == Position.MID:
            tact_overall = _weighted(
                (positioning_score, 0.3),
                (pressing_score, 0.4),
                (coverage_score, 0.3),
            )
        else:  # FWD
            tact_overall = _weighted(
                (positioning_score, 0.5),
                (pressing_score, 0.3),
                (coverage_score, 0.2),
            )

        tactical = TacticalRating(
            positioning=min(positioning_score, 100.0),
            pressing=pressing_score,
            coverage=min(coverage_score, 100.0),
            overall=tact_overall,
        )

        # ----------------------------------------------------------------
        # OVERALL — взвешенная комбинация Physical + Tactical
        # Technical добавим когда будет Event Detection
        # ----------------------------------------------------------------
        if position in (Position.GK, Position.DEF):
            overall = _weighted(
                (physical.overall, 0.4),
                (tactical.overall, 0.6),
            )
        elif position == Position.MID:
            overall = _weighted(
                (physical.overall, 0.5),
                (tactical.overall, 0.5),
            )
        else:
            overall = _weighted(
                (physical.overall, 0.6),
                (tactical.overall, 0.4),
            )

        return PlayerRating(
            player_id=phys.player_id,
            position=position,
            physical=physical,
            tactical=tactical,
            overall=min(overall, 99.0),
        )

    def calculate_all(
        self,
        stats: dict[int, tuple[PhysicalStats, PositionalStats]],
        positions: dict[int, Position] | None = None,
    ) -> list[PlayerRating]:
        """Рассчитать рейтинги для всех игроков."""
        ratings = []
        for pid, (phys, pos) in stats.items():
            position = (positions or {}).get(pid, Position.UNKNOWN)
            ratings.append(self.calculate(phys, pos, position))
        return sorted(ratings, key=lambda r: r.overall, reverse=True)
