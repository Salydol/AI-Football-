"""
CompactnessAnalyzer — измеряет компактность командного блока.

Метрики:
  - width_m     : Ширина блока (max_y - min_y) — насколько широко расставлены
  - depth_m     : Глубина блока (max_x - min_x) — насколько растянуты по длине
  - area_m2     : Площадь выпуклой оболочки игроков (convex hull area)
  - centroid_x  : X-центр тяжести команды
  - centroid_y  : Y-центр тяжести команды
  - inter_team_distance : Дистанция между центрами тяжести двух команд

Интерпретация:
  - Малая площадь + малая глубина → компактный оборонительный блок
  - Большая ширина → атакующий стиль, игра в ширину
  - Малое inter_team_distance → высокий прессинг, обе команды рядом
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fie.tracking.pipeline import TrackedPlayer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TeamShape:
    """Геометрические характеристики расположения команды."""
    team: str
    centroid_x: float
    centroid_y: float
    width: float       # max_y - min_y
    depth: float       # max_x - min_x
    area: float        # площадь convex hull (0 если < 3 игроков)
    compactness: float # 0..1 (1=очень компактно)
    player_count: int

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "centroid_x": round(self.centroid_x, 1),
            "centroid_y": round(self.centroid_y, 1),
            "width": round(self.width, 1),
            "depth": round(self.depth, 1),
            "area": round(self.area, 1),
            "compactness": round(self.compactness, 3),
            "player_count": self.player_count,
        }


@dataclass
class CompactnessSnapshot:
    """Компактность обеих команд + взаимное расположение."""
    left: TeamShape | None
    right: TeamShape | None
    inter_team_distance: float | None   # дистанция между центрами
    teams_spread: float | None          # суммарная площадь обоих блоков

    def to_dict(self) -> dict:
        return {
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "inter_team_distance": (
                round(self.inter_team_distance, 1)
                if self.inter_team_distance is not None else None
            ),
            "teams_spread": (
                round(self.teams_spread, 1)
                if self.teams_spread is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CompactnessAnalyzer:
    """
    Вычисляет компактность командного блока из позиций игроков.

    Args:
        field_length: Длина поля
        field_width:  Ширина поля
        max_area:     Эталонная максимальная площадь (для нормализации compactness)
                      По умолчанию = 50% от площади поля
    """

    def __init__(
        self,
        field_length: float = 105.0,
        field_width: float = 68.0,
        max_area: float | None = None,
    ) -> None:
        self.field_length = field_length
        self.field_width = field_width
        self.max_area = max_area or (field_length * field_width * 0.5)

    def analyze(self, players: list[TrackedPlayer]) -> CompactnessSnapshot:
        """
        Вычислить компактность для текущего кадра.

        Args:
            players: Все игроки на поле (обе команды)

        Returns:
            CompactnessSnapshot с метриками левой и правой команды
        """
        mid = self.field_length / 2.0

        left_players = [p for p in players if p.x < mid]
        right_players = [p for p in players if p.x >= mid]

        left_shape = self._compute_shape(left_players, "left") if len(left_players) >= 3 else None
        right_shape = self._compute_shape(right_players, "right") if len(right_players) >= 3 else None

        # Расстояние между центрами команд
        inter_dist = None
        if left_shape and right_shape:
            inter_dist = math.hypot(
                left_shape.centroid_x - right_shape.centroid_x,
                left_shape.centroid_y - right_shape.centroid_y,
            )

        # Суммарная площадь
        teams_spread = None
        if left_shape and right_shape:
            teams_spread = left_shape.area + right_shape.area

        return CompactnessSnapshot(
            left=left_shape,
            right=right_shape,
            inter_team_distance=inter_dist,
            teams_spread=teams_spread,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_shape(
        self,
        team_players: list[TrackedPlayer],
        team: str,
    ) -> TeamShape:
        xs = np.array([p.x for p in team_players])
        ys = np.array([p.y for p in team_players])

        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        width = float(np.max(ys) - np.min(ys))
        depth = float(np.max(xs) - np.min(xs))

        # Convex hull area
        area = self._convex_hull_area(xs, ys)

        # Компактность: меньше площадь → более компактны
        compactness = max(0.0, 1.0 - area / max(self.max_area, 1.0))

        return TeamShape(
            team=team,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            width=width,
            depth=depth,
            area=area,
            compactness=compactness,
            player_count=len(team_players),
        )

    @staticmethod
    def _convex_hull_area(xs: np.ndarray, ys: np.ndarray) -> float:
        """
        Вычислить площадь выпуклой оболочки методом Shoelace.
        Возвращает 0 если точек меньше 3.
        """
        n = len(xs)
        if n < 3:
            return 0.0

        points = np.column_stack([xs, ys])

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return float(hull.volume)  # в 2D volume = area
        except Exception:
            # Fallback: простой bbox * 0.7
            return float((xs.max() - xs.min()) * (ys.max() - ys.min()) * 0.7)
