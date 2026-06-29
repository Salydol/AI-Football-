"""
Scouting Radar — скаутинг и поиск похожих игроков.

Возможности:
  - ScoutingRadar: строит радар-чарт из 8 метрик игрока
  - PlayerSimilarityEngine: находит похожих игроков по паспортам
  - TargetProfileMatcher: сравнивает игрока с целевым профилем (напр. "нужен быстрый вингер")

Метрики радара (все нормализованы 0..100):
  pace            — максимальная скорость
  stamina         — дистанция
  sprint_power    — количество спринтов
  pressing        — интенсивность прессинга
  positioning     — позиционирование (avg_x нормализованный)
  consistency     — стабильность рейтинга
  form            — текущая форма (последние 3 матча)
  potential       — Hidden Talent Index

Использование:
    engine = PlayerSimilarityEngine(save_dir='data/passports')
    similar = engine.find_similar(player_id=7, top_k=5)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Radar Chart
# ---------------------------------------------------------------------------

@dataclass
class RadarProfile:
    """8-метричный профиль игрока для радар-чарта."""
    player_id: int
    position: str

    pace: float          # макс скорость (0..100)
    stamina: float       # дистанция (0..100)
    sprint_power: float  # спринты (0..100)
    pressing: float      # прессинг (0..100)
    positioning: float   # позиционирование (0..100)
    consistency: float   # стабильность (0..100)
    form: float          # форма (0..100)
    potential: float     # потенциал (0..100)

    overall: float       # средний по всем метрикам

    def to_dict(self) -> dict:
        return asdict(self)

    def to_vector(self) -> list[float]:
        return [
            self.pace, self.stamina, self.sprint_power, self.pressing,
            self.positioning, self.consistency, self.form, self.potential,
        ]

    def distance(self, other: "RadarProfile") -> float:
        a = np.array(self.to_vector())
        b = np.array(other.to_vector())
        return float(np.linalg.norm(a - b))

    def similarity(self, other: "RadarProfile") -> float:
        max_dist = np.sqrt(8) * 100
        return max(0.0, 1.0 - self.distance(other) / max_dist)


# ---------------------------------------------------------------------------
# Target profile для поиска
# ---------------------------------------------------------------------------

@dataclass
class TargetProfile:
    """
    Целевой профиль скаутинга — чего ищем.

    Каждая метрика: (min_value, weight)
    weight = 0 → не важно, weight = 1 → высокий приоритет.
    """
    pace_min: float = 0.0
    pace_weight: float = 0.0
    stamina_min: float = 0.0
    stamina_weight: float = 0.0
    sprint_power_min: float = 0.0
    sprint_power_weight: float = 0.0
    pressing_min: float = 0.0
    pressing_weight: float = 0.0
    positioning_min: float = 0.0
    positioning_weight: float = 0.0
    consistency_min: float = 0.0
    consistency_weight: float = 0.0
    form_min: float = 0.0
    form_weight: float = 0.0
    potential_min: float = 0.0
    potential_weight: float = 0.0
    position_filter: str | None = None  # "WNG", "ST", etc.


@dataclass
class ScoutingResult:
    """Результат поиска одного игрока."""
    player_id: int
    position: str
    radar: RadarProfile
    match_score: float        # 0..100 соответствие цели
    strengths: list[str]
    missing: list[str]        # чего не хватает до целевого профиля

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "position": self.position,
            "match_score": round(self.match_score, 1),
            "strengths": self.strengths,
            "missing": self.missing,
            "radar": self.radar.to_dict(),
        }


# ---------------------------------------------------------------------------
# ScoutingRadar — строит профиль из паспорта
# ---------------------------------------------------------------------------

# Позиционные максимумы для нормализации (elite level)
_PACE_MAX = {"GK": 22, "CB": 32, "FB": 35, "CDM": 31, "CM": 32, "CAM": 33, "WNG": 36, "ST": 35, "UNKNOWN": 33}
_DIST_MAX = {"GK": 6, "CB": 11, "FB": 13, "CDM": 13, "CM": 13, "CAM": 12, "WNG": 12, "ST": 11, "UNKNOWN": 11}
_SPRINT_MAX = {"GK": 8, "CB": 20, "FB": 50, "CDM": 30, "CM": 35, "CAM": 40, "WNG": 55, "ST": 50, "UNKNOWN": 35}


def build_radar_from_passport(passport_data: dict) -> RadarProfile | None:
    """
    Строит RadarProfile из данных паспорта (dict от PlayerPassport.get_profile().to_dict()).
    """
    pid = passport_data.get("player_id", 0)
    pos = passport_data.get("position", "UNKNOWN")
    matches = passport_data.get("match_history", [])

    if not matches:
        return None

    max_speed = passport_data.get("avg_max_speed", 0)
    avg_dist = passport_data.get("avg_distance_km", 0)
    avg_sprint = passport_data.get("avg_sprint_count", 0)
    avg_rating = passport_data.get("avg_overall_rating", 0)
    talent = passport_data.get("talent_index", 0)

    # Нормализация по позиционным максимумам
    pace = min(max_speed / max(_PACE_MAX.get(pos, 33), 1) * 100, 100)
    stamina = min(avg_dist / max(_DIST_MAX.get(pos, 11), 1) * 100, 100)
    sprint_power = min(avg_sprint / max(_SPRINT_MAX.get(pos, 35), 1) * 100, 100)

    # Прессинг из последних матчей (если есть поле pressing_intensity)
    pressing = 50.0  # placeholder

    # Позиционирование: нормализованный avg_x
    avg_x_vals = [m.get("avg_x", 0) for m in matches if m.get("avg_x", 0) > 0]
    positioning = float(np.mean(avg_x_vals)) / 105 * 100 if avg_x_vals else 50.0

    # Стабильность: 1 - std(rating) / mean(rating)
    ratings = [m.get("overall_rating", avg_rating) for m in matches]
    if len(ratings) >= 2:
        consistency = max(0, 100 - float(np.std(ratings) / max(np.mean(ratings), 1) * 100))
    else:
        consistency = 50.0

    # Форма: средний рейтинг последних 3 матчей
    recent = matches[-3:]
    form = float(np.mean([m.get("overall_rating", avg_rating) for m in recent]))

    # Потенциал
    potential = talent

    overall = float(np.mean([pace, stamina, sprint_power, pressing, positioning, consistency, form, potential]))

    return RadarProfile(
        player_id=pid,
        position=pos,
        pace=round(pace, 1),
        stamina=round(stamina, 1),
        sprint_power=round(sprint_power, 1),
        pressing=round(pressing, 1),
        positioning=round(positioning, 1),
        consistency=round(consistency, 1),
        form=round(form, 1),
        potential=round(potential, 1),
        overall=round(overall, 1),
    )


# ---------------------------------------------------------------------------
# PlayerSimilarityEngine
# ---------------------------------------------------------------------------

class PlayerSimilarityEngine:
    """
    Находит похожих игроков на основе RadarProfile.

    Args:
        save_dir:   Директория с паспортами (data/passports/)
    """

    def __init__(self, save_dir: str | Path = "data/passports") -> None:
        self.save_dir = Path(save_dir)
        self._profiles: dict[int, RadarProfile] = {}

    def _load_all(self) -> None:
        """Загрузить все паспорта из директории."""
        self._profiles.clear()
        for path in self.save_dir.glob("player_*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                matches = data.get("matches", [])
                if not matches:
                    continue
                # Строим упрощённый профиль из сырых данных
                profile_data = self._passport_raw_to_profile_dict(data)
                radar = build_radar_from_passport(profile_data)
                if radar:
                    self._profiles[radar.player_id] = radar
            except Exception:
                continue

    @staticmethod
    def _passport_raw_to_profile_dict(raw: dict) -> dict:
        """Конвертирует сырой JSON паспорта в формат get_profile().to_dict()."""
        matches = raw.get("matches", [])
        if not matches:
            return {}
        avg_speed = float(np.mean([m.get("max_speed_kmh", 0) for m in matches]))
        avg_dist = float(np.mean([m.get("distance_km", 0) for m in matches]))
        avg_sprint = float(np.mean([m.get("sprint_count", 0) for m in matches]))
        avg_rating = float(np.mean([m.get("overall_rating", 0) for m in matches]))

        # Talent index (простая формула)
        talent = min(avg_rating * 0.6 + min(avg_speed / 35.0, 1.0) * 10, 100.0)

        return {
            "player_id": raw.get("player_id", 0),
            "position": raw.get("position", "UNKNOWN"),
            "avg_max_speed": avg_speed,
            "avg_distance_km": avg_dist,
            "avg_sprint_count": avg_sprint,
            "avg_overall_rating": avg_rating,
            "talent_index": talent,
            "match_history": matches,
        }

    def find_similar(
        self,
        player_id: int,
        top_k: int = 5,
        position_filter: str | None = None,
    ) -> list[dict]:
        """
        Найти top_k игроков похожих на player_id.

        Returns:
            Список dict с player_id, similarity, radar
        """
        self._load_all()
        target = self._profiles.get(player_id)
        if not target:
            return []

        results = []
        for pid, profile in self._profiles.items():
            if pid == player_id:
                continue
            if position_filter and profile.position != position_filter:
                continue
            sim = target.similarity(profile)
            results.append({
                "player_id": pid,
                "position": profile.position,
                "similarity": round(sim, 3),
                "radar": profile.to_dict(),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def find_by_target(
        self,
        target: TargetProfile,
        top_k: int = 10,
    ) -> list[ScoutingResult]:
        """
        Найти игроков под целевой профиль.

        Учитывает минимальные значения и веса метрик.
        """
        self._load_all()
        results = []

        metric_map = [
            ("pace", target.pace_min, target.pace_weight),
            ("stamina", target.stamina_min, target.stamina_weight),
            ("sprint_power", target.sprint_power_min, target.sprint_power_weight),
            ("pressing", target.pressing_min, target.pressing_weight),
            ("positioning", target.positioning_min, target.positioning_weight),
            ("consistency", target.consistency_min, target.consistency_weight),
            ("form", target.form_min, target.form_weight),
            ("potential", target.potential_min, target.potential_weight),
        ]

        for pid, profile in self._profiles.items():
            if target.position_filter and profile.position != target.position_filter:
                continue

            vals = profile.to_vector()
            metric_names = ["pace", "stamina", "sprint_power", "pressing",
                            "positioning", "consistency", "form", "potential"]

            # Проверка минимальных значений
            meets_min = all(
                vals[i] >= req_min
                for i, (_, req_min, _) in enumerate(metric_map)
                if req_min > 0
            )
            if not meets_min:
                continue

            # Взвешенная оценка
            total_weight = sum(w for _, _, w in metric_map) or 1.0
            score = sum(
                vals[i] * w / total_weight
                for i, (_, _, w) in enumerate(metric_map)
            )

            strengths = [
                metric_names[i]
                for i, (_, req_min, w) in enumerate(metric_map)
                if w > 0 and vals[i] >= req_min * 1.1
            ]
            missing = [
                metric_names[i]
                for i, (_, req_min, _) in enumerate(metric_map)
                if req_min > 0 and vals[i] < req_min
            ]

            results.append(ScoutingResult(
                player_id=pid,
                position=profile.position,
                radar=profile,
                match_score=round(score, 1),
                strengths=strengths,
                missing=missing,
            ))

        results.sort(key=lambda r: r.match_score, reverse=True)
        return results[:top_k]

    def get_radar(self, player_id: int) -> RadarProfile | None:
        """Получить RadarProfile для игрока."""
        self._load_all()
        return self._profiles.get(player_id)

    def compare_players(self, player_id_1: int, player_id_2: int) -> dict:
        """Сравнить двух игроков."""
        self._load_all()
        p1 = self._profiles.get(player_id_1)
        p2 = self._profiles.get(player_id_2)
        if not p1 or not p2:
            return {"error": "One or both players not found"}
        return {
            "similarity": round(p1.similarity(p2), 3),
            "player_1": p1.to_dict(),
            "player_2": p2.to_dict(),
            "diff": {
                k: round(getattr(p1, k) - getattr(p2, k), 1)
                for k in ["pace", "stamina", "sprint_power", "pressing",
                           "consistency", "form", "potential"]
            },
        }
