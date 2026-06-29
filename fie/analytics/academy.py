"""
Academy Progress Tracker — отслеживание прогресса молодых игроков.

Функции:
  - Возрастные нормы (U15, U17, U19, U21, U23) для физических метрик
  - Сезонный прогресс (сравнение начало vs конец сезона)
  - Development Score — комплексная оценка развития
  - First Team Readiness — % готовности к переходу в основу
  - Сравнение с одногодками из академии
  - Рекомендации по развитию

Хранение: data/academy/{player_id}.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Age groups & benchmarks
# ---------------------------------------------------------------------------

class AgeGroup(str, Enum):
    U15 = "U15"
    U17 = "U17"
    U19 = "U19"
    U21 = "U21"
    U23 = "U23"
    SENIOR = "SENIOR"


def get_age_group(age: int) -> AgeGroup:
    if age < 15:
        return AgeGroup.U15
    elif age < 17:
        return AgeGroup.U15
    elif age < 19:
        return AgeGroup.U17
    elif age < 21:
        return AgeGroup.U19
    elif age < 23:
        return AgeGroup.U21
    elif age < 25:
        return AgeGroup.U23
    else:
        return AgeGroup.SENIOR


# Физические нормы для каждой возрастной группы (elite academy level)
AGE_BENCHMARKS: dict[str, dict[str, float]] = {
    "U15": {
        "distance_km": 7.5,
        "max_speed_kmh": 27.0,
        "sprint_count": 15,
        "overall_rating": 55.0,
        "physical_rating": 55.0,
        "tactical_rating": 50.0,
    },
    "U17": {
        "distance_km": 9.0,
        "max_speed_kmh": 29.5,
        "sprint_count": 22,
        "overall_rating": 62.0,
        "physical_rating": 63.0,
        "tactical_rating": 58.0,
    },
    "U19": {
        "distance_km": 10.0,
        "max_speed_kmh": 31.5,
        "sprint_count": 28,
        "overall_rating": 68.0,
        "physical_rating": 70.0,
        "tactical_rating": 65.0,
    },
    "U21": {
        "distance_km": 10.8,
        "max_speed_kmh": 33.0,
        "sprint_count": 33,
        "overall_rating": 73.0,
        "physical_rating": 75.0,
        "tactical_rating": 70.0,
    },
    "U23": {
        "distance_km": 11.2,
        "max_speed_kmh": 34.0,
        "sprint_count": 36,
        "overall_rating": 76.0,
        "physical_rating": 78.0,
        "tactical_rating": 73.0,
    },
    "SENIOR": {
        "distance_km": 11.5,
        "max_speed_kmh": 34.5,
        "sprint_count": 38,
        "overall_rating": 80.0,
        "physical_rating": 80.0,
        "tactical_rating": 78.0,
    },
}

# Порог готовности к первой команде (% от senior benchmark)
FIRST_TEAM_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Session record
# ---------------------------------------------------------------------------

@dataclass
class AcademySession:
    """Данные одной тренировки/матча молодого игрока."""
    session_id: str
    date: str                   # YYYY-MM-DD
    session_type: str           # "match" / "training"
    age_at_session: int

    # Физика
    distance_km: float
    max_speed_kmh: float
    sprint_count: int
    high_accel_count: int

    # Рейтинги
    physical_rating: float
    tactical_rating: float
    overall_rating: float

    # Доп. данные
    coach_note: str = ""
    minutes_played: float = 90.0
    fatigue_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AcademySession":
        return cls(**d)


# ---------------------------------------------------------------------------
# Development metrics
# ---------------------------------------------------------------------------

@dataclass
class SeasonProgress:
    """Прогресс за один сезон."""
    season: str           # "2024-25"
    sessions_count: int
    avg_overall: float
    avg_physical: float
    avg_tactical: float
    avg_distance: float
    avg_speed: float
    avg_sprints: float
    start_rating: float   # первые 3 матча
    end_rating: float     # последние 3 матча
    improvement_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgeComparison:
    """Сравнение с возрастным эталоном."""
    age_group: str
    distance_pct: float      # % от нормы
    speed_pct: float
    sprint_pct: float
    overall_pct: float
    physical_pct: float
    tactical_pct: float
    meets_benchmark: bool    # соответствует ли норме

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FirstTeamReadiness:
    """Оценка готовности к переходу в основную команду."""
    readiness_pct: float      # 0..100
    readiness_label: str      # "not_ready" / "developing" / "close" / "ready"
    distance_ready: bool
    speed_ready: bool
    sprint_ready: bool
    rating_ready: bool
    estimated_seasons: int    # сколько сезонов до готовности
    recommendations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AcademyProfile:
    """Полный профиль академиста."""
    player_id: int
    name: str
    age: int
    age_group: str
    position: str
    seasons_in_academy: int

    # Текущие средние
    avg_overall: float
    avg_physical: float
    avg_tactical: float
    avg_distance: float
    avg_max_speed: float

    # Прогресс
    season_progress: list[SeasonProgress]
    career_trend: str          # "rapid_growth" / "steady" / "plateau" / "decline"
    development_score: float   # 0..100 (итоговая оценка развития)

    # Сравнение
    age_comparison: AgeComparison
    first_team_readiness: FirstTeamReadiness

    # Сильные стороны
    strengths: list[str]
    development_areas: list[str]

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "age": self.age,
            "age_group": self.age_group,
            "position": self.position,
            "seasons_in_academy": self.seasons_in_academy,
            "avg_overall": round(self.avg_overall, 1),
            "avg_physical": round(self.avg_physical, 1),
            "avg_tactical": round(self.avg_tactical, 1),
            "avg_distance": round(self.avg_distance, 2),
            "avg_max_speed": round(self.avg_max_speed, 1),
            "career_trend": self.career_trend,
            "development_score": round(self.development_score, 1),
            "age_comparison": self.age_comparison.to_dict(),
            "first_team_readiness": self.first_team_readiness.to_dict(),
            "season_progress": [s.to_dict() for s in self.season_progress],
            "strengths": self.strengths,
            "development_areas": self.development_areas,
        }


# ---------------------------------------------------------------------------
# AcademyTracker
# ---------------------------------------------------------------------------

class AcademyTracker:
    """
    Отслеживает прогресс молодого игрока в академии.

    Args:
        player_id:  ID игрока
        name:       Имя игрока
        position:   Позиция (GK/CB/FB/CDM/CM/CAM/WNG/ST)
        save_dir:   Директория для хранения данных
    """

    def __init__(
        self,
        player_id: int,
        name: str = "",
        position: str = "UNKNOWN",
        save_dir: str | Path = "data/academy",
    ) -> None:
        self.player_id = player_id
        self.name = name or f"Player {player_id}"
        self.position = position
        self.save_dir = Path(save_dir)
        self._sessions: list[AcademySession] = []

    @classmethod
    def load(cls, path: str | Path) -> "AcademyTracker":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tracker = cls(
            player_id=data["player_id"],
            name=data.get("name", ""),
            position=data.get("position", "UNKNOWN"),
            save_dir=path.parent,
        )
        tracker._sessions = [AcademySession.from_dict(s) for s in data.get("sessions", [])]
        return tracker

    @classmethod
    def load_or_create(
        cls,
        player_id: int,
        name: str = "",
        position: str = "UNKNOWN",
        save_dir: str | Path = "data/academy",
    ) -> "AcademyTracker":
        path = Path(save_dir) / f"player_{player_id}.json"
        if path.exists():
            return cls.load(path)
        return cls(player_id=player_id, name=name, position=position, save_dir=save_dir)

    def add_session(self, session: AcademySession) -> None:
        """Добавить сессию (матч/тренировка)."""
        self._sessions = [s for s in self._sessions if s.session_id != session.session_id]
        self._sessions.append(session)
        self._sessions.sort(key=lambda s: s.date)

    def save(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"player_{self.player_id}.json"
        data = {
            "player_id": self.player_id,
            "name": self.name,
            "position": self.position,
            "updated_at": datetime.now().isoformat(),
            "sessions": [s.to_dict() for s in self._sessions],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def get_profile(self) -> AcademyProfile | None:
        if not self._sessions:
            return None

        sessions = self._sessions
        ages = [s.age_at_session for s in sessions]
        current_age = max(ages)
        age_group = get_age_group(current_age)
        seasons_in_academy = self._count_seasons()

        # Средние показатели
        avg_overall = float(np.mean([s.overall_rating for s in sessions]))
        avg_physical = float(np.mean([s.physical_rating for s in sessions]))
        avg_tactical = float(np.mean([s.tactical_rating for s in sessions]))
        avg_distance = float(np.mean([s.distance_km for s in sessions]))
        avg_speed = float(np.mean([s.max_speed_kmh for s in sessions]))

        # Прогресс по сезонам
        season_progress = self._compute_season_progress()

        # Карьерный тренд
        career_trend = self._compute_career_trend(season_progress)

        # Development Score
        dev_score = self._compute_development_score(
            avg_overall, avg_physical, avg_tactical, age_group, season_progress
        )

        # Сравнение с возрастом
        age_comp = self._compare_with_age(avg_distance, avg_speed,
                                           float(np.mean([s.sprint_count for s in sessions])),
                                           avg_overall, avg_physical, avg_tactical,
                                           age_group)

        # Готовность к первой команде
        readiness = self._compute_readiness(avg_distance, avg_speed,
                                             float(np.mean([s.sprint_count for s in sessions])),
                                             avg_overall)

        # Сильные стороны и зоны развития
        strengths, areas = self._analyze_development(avg_physical, avg_tactical,
                                                       avg_speed, avg_distance, dev_score)

        return AcademyProfile(
            player_id=self.player_id,
            name=self.name,
            age=current_age,
            age_group=age_group.value,
            position=self.position,
            seasons_in_academy=seasons_in_academy,
            avg_overall=avg_overall,
            avg_physical=avg_physical,
            avg_tactical=avg_tactical,
            avg_distance=avg_distance,
            avg_max_speed=avg_speed,
            season_progress=season_progress,
            career_trend=career_trend,
            development_score=dev_score,
            age_comparison=age_comp,
            first_team_readiness=readiness,
            strengths=strengths,
            development_areas=areas,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _count_seasons(self) -> int:
        seasons = set()
        for s in self._sessions:
            year = int(s.date[:4])
            month = int(s.date[5:7])
            season = f"{year}-{year+1}" if month >= 7 else f"{year-1}-{year}"
            seasons.add(season)
        return max(len(seasons), 1)

    def _get_season(self, date: str) -> str:
        year = int(date[:4])
        month = int(date[5:7])
        return f"{year}-{year+1}" if month >= 7 else f"{year-1}-{year}"

    def _compute_season_progress(self) -> list[SeasonProgress]:
        by_season: dict[str, list[AcademySession]] = {}
        for s in self._sessions:
            season = self._get_season(s.date)
            by_season.setdefault(season, []).append(s)

        result = []
        for season, sess in sorted(by_season.items()):
            ratings = [s.overall_rating for s in sess]
            start = float(np.mean(ratings[:3])) if len(ratings) >= 3 else ratings[0]
            end = float(np.mean(ratings[-3:])) if len(ratings) >= 3 else ratings[-1]
            improvement = (end - start) / max(start, 1) * 100

            result.append(SeasonProgress(
                season=season,
                sessions_count=len(sess),
                avg_overall=round(float(np.mean(ratings)), 1),
                avg_physical=round(float(np.mean([s.physical_rating for s in sess])), 1),
                avg_tactical=round(float(np.mean([s.tactical_rating for s in sess])), 1),
                avg_distance=round(float(np.mean([s.distance_km for s in sess])), 2),
                avg_speed=round(float(np.mean([s.max_speed_kmh for s in sess])), 1),
                avg_sprints=round(float(np.mean([s.sprint_count for s in sess])), 1),
                start_rating=round(start, 1),
                end_rating=round(end, 1),
                improvement_pct=round(improvement, 1),
            ))
        return result

    def _compute_career_trend(self, season_progress: list[SeasonProgress]) -> str:
        if len(season_progress) < 2:
            return "insufficient_data"
        improvements = [s.improvement_pct for s in season_progress]
        avg_imp = float(np.mean(improvements))
        if avg_imp > 10:
            return "rapid_growth"
        elif avg_imp > 3:
            return "steady"
        elif avg_imp > -3:
            return "plateau"
        else:
            return "decline"

    def _compute_development_score(
        self,
        avg_overall: float,
        avg_physical: float,
        avg_tactical: float,
        age_group: AgeGroup,
        season_progress: list[SeasonProgress],
    ) -> float:
        bench = AGE_BENCHMARKS[age_group.value]

        # Соответствие возрастной норме
        age_match = min(avg_overall / max(bench["overall_rating"], 1) * 100, 120)

        # Тренд улучшения
        if season_progress:
            trend_bonus = max(0, float(np.mean([s.improvement_pct for s in season_progress])) * 0.5)
        else:
            trend_bonus = 0.0

        score = min(age_match * 0.7 + trend_bonus * 0.3, 100.0)
        return round(score, 1)

    def _compare_with_age(
        self,
        avg_dist: float,
        avg_speed: float,
        avg_sprint: float,
        avg_overall: float,
        avg_physical: float,
        avg_tactical: float,
        age_group: AgeGroup,
    ) -> AgeComparison:
        bench = AGE_BENCHMARKS[age_group.value]
        dist_pct = min(avg_dist / max(bench["distance_km"], 0.1) * 100, 150)
        speed_pct = min(avg_speed / max(bench["max_speed_kmh"], 0.1) * 100, 150)
        sprint_pct = min(avg_sprint / max(bench["sprint_count"], 0.1) * 100, 150)
        overall_pct = min(avg_overall / max(bench["overall_rating"], 0.1) * 100, 150)
        phys_pct = min(avg_physical / max(bench["physical_rating"], 0.1) * 100, 150)
        tact_pct = min(avg_tactical / max(bench["tactical_rating"], 0.1) * 100, 150)

        meets = all(p >= 80 for p in [dist_pct, speed_pct, overall_pct])

        return AgeComparison(
            age_group=age_group.value,
            distance_pct=round(dist_pct, 1),
            speed_pct=round(speed_pct, 1),
            sprint_pct=round(sprint_pct, 1),
            overall_pct=round(overall_pct, 1),
            physical_pct=round(phys_pct, 1),
            tactical_pct=round(tact_pct, 1),
            meets_benchmark=meets,
        )

    def _compute_readiness(
        self,
        avg_dist: float,
        avg_speed: float,
        avg_sprint: float,
        avg_overall: float,
    ) -> FirstTeamReadiness:
        senior = AGE_BENCHMARKS["SENIOR"]
        thr = FIRST_TEAM_THRESHOLD

        dist_ready = avg_dist >= senior["distance_km"] * thr
        speed_ready = avg_speed >= senior["max_speed_kmh"] * thr
        sprint_ready = avg_sprint >= senior["sprint_count"] * thr
        rating_ready = avg_overall >= senior["overall_rating"] * thr

        ready_count = sum([dist_ready, speed_ready, sprint_ready, rating_ready])
        readiness_pct = ready_count / 4 * 100

        if readiness_pct >= 90:
            label = "ready"
        elif readiness_pct >= 70:
            label = "close"
        elif readiness_pct >= 40:
            label = "developing"
        else:
            label = "not_ready"

        # Оценка: сколько сезонов до готовности
        gap = senior["overall_rating"] * thr - avg_overall
        if gap <= 0:
            estimated = 0
        else:
            estimated = max(1, int(gap / 4))  # ~4 балла роста в сезон

        recs = []
        if not dist_ready:
            recs.append(f"Increase match distance to {senior['distance_km'] * thr:.1f}+ km/90min")
        if not speed_ready:
            recs.append(f"Develop top speed to {senior['max_speed_kmh'] * thr:.1f}+ km/h")
        if not sprint_ready:
            recs.append(f"Build sprint endurance to {int(senior['sprint_count'] * thr)}+ per match")
        if not rating_ready:
            recs.append(f"Raise overall rating to {senior['overall_rating'] * thr:.0f}+")
        if not recs:
            recs.append("Ready for first team consideration. Recommend loan or integration.")

        return FirstTeamReadiness(
            readiness_pct=round(readiness_pct, 1),
            readiness_label=label,
            distance_ready=dist_ready,
            speed_ready=speed_ready,
            sprint_ready=sprint_ready,
            rating_ready=rating_ready,
            estimated_seasons=estimated,
            recommendations=recs,
        )

    def _analyze_development(
        self,
        avg_physical: float,
        avg_tactical: float,
        avg_speed: float,
        avg_distance: float,
        dev_score: float,
    ) -> tuple[list[str], list[str]]:
        strengths = []
        areas = []

        if avg_speed >= 30:
            strengths.append("Exceptional pace for age group")
        if avg_distance >= 10.0:
            strengths.append("High work rate and stamina")
        if avg_physical >= 70:
            strengths.append("Strong physical development")
        if avg_tactical >= 70:
            strengths.append("Advanced tactical understanding")
        if dev_score >= 80:
            strengths.append("Ahead of development curve")

        if avg_speed < 25:
            areas.append("Speed development — focus on sprint training")
        if avg_distance < 8.5:
            areas.append("Stamina — increase high-intensity interval work")
        if avg_physical < 60:
            areas.append("Physical conditioning — strength & conditioning program")
        if avg_tactical < 60:
            areas.append("Tactical awareness — positional coaching required")
        if dev_score < 50:
            areas.append("Overall development below age benchmark — intensive program needed")

        return (
            strengths or ["Consistent effort across sessions"],
            areas or ["Continue current development path"],
        )
