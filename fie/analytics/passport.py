"""
PlayerPassport — цифровой паспорт игрока.

Хранит данные по нескольким матчам и строит:
  - физический профиль (скорость, дистанция, спринты)
  - тактический профиль (позиционирование, прессинг)
  - прогресс за сезон
  - сравнение с позиционным эталоном
  - сильные и слабые стороны
  - Hidden Talent Index (потенциал)

Хранилище: JSON-файл на диске (один файл = один игрок).
При каждом матче данные дописываются.

Использование:
    passport = PlayerPassport.load("data/passports/player_7.json")
    passport.add_match(match_data)
    passport.save()
    profile = passport.get_profile()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Position benchmarks
# ---------------------------------------------------------------------------

class DetailedPosition(str, Enum):
    GK = "GK"           # вратарь
    CB = "CB"           # центральный защитник
    FB = "FB"           # фулбек (LB/RB)
    CDM = "CDM"         # опорный полузащитник
    CM = "CM"           # центральный полузащитник
    CAM = "CAM"         # атакующий полузащитник
    WNG = "WNG"         # вингер (LW/RW)
    ST = "ST"           # нападающий
    UNKNOWN = "UNKNOWN"


# Эталонные значения для каждой позиции (elite level, per 90 min)
POSITION_BENCHMARKS: dict[str, dict[str, float]] = {
    "GK": {
        "distance_km": 5.5,
        "max_speed_kmh": 22.0,
        "sprint_count": 5,
        "pressing_intensity": 0.05,
        "positioning_score": 90.0,
    },
    "CB": {
        "distance_km": 9.5,
        "max_speed_kmh": 30.0,
        "sprint_count": 15,
        "pressing_intensity": 0.15,
        "positioning_score": 80.0,
        "duel_score": 75.0,
    },
    "FB": {
        "distance_km": 11.5,
        "max_speed_kmh": 33.0,
        "sprint_count": 35,
        "pressing_intensity": 0.25,
        "crossing_score": 65.0,
    },
    "CDM": {
        "distance_km": 12.0,
        "max_speed_kmh": 30.0,
        "sprint_count": 20,
        "pressing_intensity": 0.30,
        "positioning_score": 75.0,
    },
    "CM": {
        "distance_km": 12.5,
        "max_speed_kmh": 31.0,
        "sprint_count": 25,
        "pressing_intensity": 0.28,
        "pass_involvement": 80.0,
    },
    "CAM": {
        "distance_km": 11.0,
        "max_speed_kmh": 32.0,
        "sprint_count": 30,
        "pressing_intensity": 0.20,
        "chance_creation": 70.0,
    },
    "WNG": {
        "distance_km": 11.0,
        "max_speed_kmh": 35.0,
        "sprint_count": 45,
        "pressing_intensity": 0.20,
        "dribble_score": 65.0,
    },
    "ST": {
        "distance_km": 10.0,
        "max_speed_kmh": 34.0,
        "sprint_count": 40,
        "pressing_intensity": 0.15,
        "xg_score": 0.5,
    },
    "UNKNOWN": {
        "distance_km": 10.5,
        "max_speed_kmh": 31.0,
        "sprint_count": 25,
        "pressing_intensity": 0.20,
    },
}


# ---------------------------------------------------------------------------
# Match record
# ---------------------------------------------------------------------------

@dataclass
class MatchRecord:
    """Данные одного матча для паспорта."""
    match_id: str
    date: str
    opponent: str
    duration_minutes: float

    # Физика
    distance_km: float
    max_speed_kmh: float
    avg_speed_kmh: float
    sprint_count: int
    high_accel_count: int

    # Рейтинги
    physical_rating: float
    tactical_rating: float
    overall_rating: float

    # Тактика
    avg_x: float            # средняя позиция по длине поля
    avg_y: float            # средняя позиция по ширине
    time_near_ball_s: float

    # Усталость
    fatigue_score: float
    injury_risk: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchRecord":
        return cls(**d)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class PositionComparison:
    """Сравнение игрока с эталоном его позиции."""
    position: str
    distance_pct: float       # % от эталона (100 = на уровне элиты)
    speed_pct: float
    sprint_pct: float
    pressing_pct: float
    overall_match_pct: float  # общее соответствие позиции
    strengths: list[str]
    weaknesses: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlayerProfile:
    """Полный профиль игрока."""
    player_id: int
    position: str
    total_matches: int

    # Средние показатели за сезон
    avg_distance_km: float
    avg_max_speed: float
    avg_sprint_count: float
    avg_overall_rating: float

    # Прогресс (последние 5 матчей vs предыдущие 5)
    progress_pct: float         # +% = растёт, -% = падает
    trend: str                  # "improving" / "stable" / "declining"

    # Сравнение с позицией
    position_comparison: PositionComparison

    # Hidden Talent Index
    talent_index: float         # 0..100 (потенциал)
    potential_rating: float     # предполагаемый пик рейтинга

    # Сильные и слабые стороны
    strengths: list[str]
    weaknesses: list[str]

    # История матчей
    match_history: list[MatchRecord]

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "position": self.position,
            "total_matches": self.total_matches,
            "avg_distance_km": round(self.avg_distance_km, 2),
            "avg_max_speed": round(self.avg_max_speed, 1),
            "avg_sprint_count": round(self.avg_sprint_count, 1),
            "avg_overall_rating": round(self.avg_overall_rating, 1),
            "progress_pct": round(self.progress_pct, 1),
            "trend": self.trend,
            "position_comparison": self.position_comparison.to_dict(),
            "talent_index": round(self.talent_index, 1),
            "potential_rating": round(self.potential_rating, 1),
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "match_history": [m.to_dict() for m in self.match_history],
        }


# ---------------------------------------------------------------------------
# Passport
# ---------------------------------------------------------------------------

class PlayerPassport:
    """
    Цифровой паспорт игрока — хранит и анализирует данные за сезон.

    Args:
        player_id:  ID игрока
        position:   Позиция (DetailedPosition)
        save_dir:   Директория для сохранения JSON
    """

    def __init__(
        self,
        player_id: int,
        position: str = "UNKNOWN",
        save_dir: str | Path = "data/passports",
    ) -> None:
        self.player_id = player_id
        self.position = position
        self.save_dir = Path(save_dir)
        self._matches: list[MatchRecord] = []

    @classmethod
    def load(cls, path: str | Path) -> "PlayerPassport":
        """Загрузить паспорт из JSON файла."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Passport not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        passport = cls(
            player_id=data["player_id"],
            position=data.get("position", "UNKNOWN"),
            save_dir=path.parent,
        )
        passport._matches = [MatchRecord.from_dict(m) for m in data.get("matches", [])]
        return passport

    @classmethod
    def load_or_create(
        cls,
        player_id: int,
        position: str = "UNKNOWN",
        save_dir: str | Path = "data/passports",
    ) -> "PlayerPassport":
        """Загрузить если существует, иначе создать новый."""
        path = Path(save_dir) / f"player_{player_id}.json"
        if path.exists():
            return cls.load(path)
        return cls(player_id=player_id, position=position, save_dir=save_dir)

    def add_match(self, record: MatchRecord) -> None:
        """Добавить данные матча в паспорт."""
        # Убрать дубль если уже есть
        self._matches = [m for m in self._matches if m.match_id != record.match_id]
        self._matches.append(record)
        # Сортировать по дате
        self._matches.sort(key=lambda m: m.date)

    def save(self) -> Path:
        """Сохранить паспорт на диск."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"player_{self.player_id}.json"
        data = {
            "player_id": self.player_id,
            "position": self.position,
            "updated_at": datetime.now().isoformat(),
            "matches": [m.to_dict() for m in self._matches],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def get_profile(self) -> PlayerProfile | None:
        """Построить полный профиль игрока."""
        if not self._matches:
            return None

        matches = self._matches
        n = len(matches)

        # Средние показатели
        avg_dist = float(np.mean([m.distance_km for m in matches]))
        avg_speed = float(np.mean([m.max_speed_kmh for m in matches]))
        avg_sprint = float(np.mean([m.sprint_count for m in matches]))
        avg_rating = float(np.mean([m.overall_rating for m in matches]))

        # Прогресс: последние 5 vs предыдущие 5
        recent = matches[-5:] if n >= 5 else matches[-max(1, n//2):]
        older = matches[:-5] if n >= 10 else matches[:max(1, n//2)]

        if older:
            recent_avg = float(np.mean([m.overall_rating for m in recent]))
            older_avg = float(np.mean([m.overall_rating for m in older]))
            progress = (recent_avg - older_avg) / max(older_avg, 1) * 100
        else:
            progress = 0.0

        if progress > 5:
            trend = "improving"
        elif progress < -5:
            trend = "declining"
        else:
            trend = "stable"

        # Сравнение с позицией
        pos_comp = self._compare_with_position(avg_dist, avg_speed, avg_sprint)

        # Сильные и слабые стороны
        strengths, weaknesses = self._analyze_strengths(avg_dist, avg_speed, avg_sprint, avg_rating)

        # Hidden Talent Index
        talent, potential = self._compute_talent_index(
            avg_rating, progress, avg_speed, avg_sprint, n
        )

        return PlayerProfile(
            player_id=self.player_id,
            position=self.position,
            total_matches=n,
            avg_distance_km=avg_dist,
            avg_max_speed=avg_speed,
            avg_sprint_count=avg_sprint,
            avg_overall_rating=avg_rating,
            progress_pct=progress,
            trend=trend,
            position_comparison=pos_comp,
            talent_index=talent,
            potential_rating=potential,
            strengths=strengths,
            weaknesses=weaknesses,
            match_history=matches[-10:],  # последние 10 матчей
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compare_with_position(
        self,
        avg_dist: float,
        avg_speed: float,
        avg_sprint: float,
    ) -> PositionComparison:
        bench = POSITION_BENCHMARKS.get(self.position, POSITION_BENCHMARKS["UNKNOWN"])

        dist_pct = min(avg_dist / max(bench["distance_km"], 0.1) * 100, 150)
        speed_pct = min(avg_speed / max(bench["max_speed_kmh"], 0.1) * 100, 150)
        sprint_pct = min(avg_sprint / max(bench["sprint_count"], 0.1) * 100, 150)
        pressing_pct = 60.0  # placeholder без данных прессинга в матче

        overall_pct = float(np.mean([dist_pct, speed_pct, sprint_pct]))

        strengths = []
        weaknesses = []

        if dist_pct >= 90:
            strengths.append("High work rate (distance)")
        elif dist_pct < 70:
            weaknesses.append("Below average distance covered")

        if speed_pct >= 95:
            strengths.append("Excellent top speed")
        elif speed_pct < 75:
            weaknesses.append("Limited top speed for position")

        if sprint_pct >= 90:
            strengths.append("Strong sprint output")
        elif sprint_pct < 65:
            weaknesses.append("Low sprint count vs position benchmark")

        return PositionComparison(
            position=self.position,
            distance_pct=round(dist_pct, 1),
            speed_pct=round(speed_pct, 1),
            sprint_pct=round(sprint_pct, 1),
            pressing_pct=round(pressing_pct, 1),
            overall_match_pct=round(overall_pct, 1),
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def _analyze_strengths(
        self,
        avg_dist: float,
        avg_speed: float,
        avg_sprint: float,
        avg_rating: float,
    ) -> tuple[list[str], list[str]]:
        strengths = []
        weaknesses = []

        if avg_speed >= 30:
            strengths.append("High top speed")
        if avg_dist >= 10.5:
            strengths.append("High work rate")
        if avg_sprint >= 35:
            strengths.append("Explosive sprint ability")
        if avg_rating >= 75:
            strengths.append("Consistently high overall rating")

        if avg_speed < 25:
            weaknesses.append("Below average pace")
        if avg_dist < 8.0:
            weaknesses.append("Low distance covered")
        if avg_sprint < 10:
            weaknesses.append("Limited explosive runs")
        if avg_rating < 55:
            weaknesses.append("Below average overall performance")

        return strengths or ["Consistent performer"], weaknesses or ["No critical weaknesses identified"]

    @staticmethod
    def _compute_talent_index(
        avg_rating: float,
        progress: float,
        avg_speed: float,
        avg_sprint: float,
        n_matches: int,
    ) -> tuple[float, float]:
        """
        Hidden Talent Index — оценка потенциала игрока.

        Учитывает:
        - текущий рейтинг
        - тренд роста
        - физические данные (скорость, спринты)
        - количество матчей (чем меньше, тем выше неопределённость)
        """
        # Базовый индекс от рейтинга
        base = avg_rating * 0.6

        # Бонус за рост
        growth_bonus = max(0, progress * 0.5)

        # Физический потенциал
        phys_bonus = (
            min(avg_speed / 35.0, 1.0) * 10 +
            min(avg_sprint / 50.0, 1.0) * 10
        )

        talent = min(base + growth_bonus + phys_bonus, 100.0)

        # Потенциальный пик = текущий + запас роста
        # Молодые игроки с трендом роста имеют больший потенциал
        growth_factor = 1.0 + max(0, progress / 100) * 0.3
        if n_matches < 10:
            growth_factor *= 1.1  # неопределённость = потенциал

        potential = min(avg_rating * growth_factor + growth_bonus, 99.0)

        return round(talent, 1), round(potential, 1)
