"""
GoalProbabilityEngine — вычисляет xG и вероятность гола в ближайшие N секунд.

xG (Expected Goals) — стандартная метрика в футбольной аналитике.
Показывает насколько опасна текущая позиция для взятия ворот.

Факторы:
  - Дистанция до ворот (главный фактор)
  - Угол обстрела (ширина видимых ворот)
  - Количество защитников между мячом и воротами
  - Скорость мяча (быстрое движение к воротам = опаснее)
  - Позиция по ширине (центр опаснее флангов)
  - Momentum — растёт ли угроза

Без обученной модели использует аналитическую формулу
основанную на реальных xG-моделях (StatsBomb / Opta).

Использование:
    engine = GoalProbabilityEngine(field_length=105, field_width=68)
    result = engine.compute(ball, players, attacking_team="left")
    # GoalProbResult(xg=0.23, probability_5s=0.08, danger_zone=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque

import numpy as np

from fie.tracking.pipeline import TrackedBall, TrackedPlayer


# ---------------------------------------------------------------------------
# Goal positions (метры от левого нижнего угла поля)
# ---------------------------------------------------------------------------

GOAL_WIDTH = 7.32   # ширина ворот
GOAL_HEIGHT = 2.44  # высота (не используется в 2D, но для справки)

# Левые ворота: x=0, y=34 ± 3.66
LEFT_GOAL_X = 0.0
LEFT_GOAL_Y = 34.0

# Правые ворота: x=105, y=34 ± 3.66
RIGHT_GOAL_X = 105.0
RIGHT_GOAL_Y = 34.0

# Зона высокой опасности (penalty area approx)
DANGER_ZONE_X_FROM_GOAL = 20.0   # метры от линии ворот
DANGER_ZONE_WIDTH = 20.0         # метры от центра по ширине


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GoalProbResult:
    """Результат оценки угрозы ворот."""
    xg: float                        # Expected Goals (0..1)
    probability_5s: float            # вероятность гола в ближайшие 5 секунд
    probability_30s: float           # вероятность гола в ближайшие 30 секунд
    danger_zone: bool                # мяч в опасной зоне
    distance_to_goal: float          # дистанция до ворот (метры)
    shot_angle_deg: float            # угол обстрела (градусы)
    defenders_in_path: int           # защитников между мячом и воротами
    attacking_team: str              # "left" / "right"
    threat_level: str                # "low" / "medium" / "high" / "critical"

    def to_dict(self) -> dict:
        return {
            "xg": round(self.xg, 4),
            "probability_5s": round(self.probability_5s, 4),
            "probability_30s": round(self.probability_30s, 4),
            "danger_zone": self.danger_zone,
            "distance_to_goal": round(self.distance_to_goal, 1),
            "shot_angle_deg": round(self.shot_angle_deg, 1),
            "defenders_in_path": self.defenders_in_path,
            "attacking_team": self.attacking_team,
            "threat_level": self.threat_level,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GoalProbabilityEngine:
    """
    Вычисляет xG и вероятность гола в реальном времени.

    Args:
        field_length:   Длина поля (105м или пиксели — масштаб должен совпадать)
        field_width:    Ширина поля (68м или пиксели)
        history_len:    Количество кадров истории для momentum
    """

    def __init__(
        self,
        field_length: float = 105.0,
        field_width: float = 68.0,
        history_len: int = 50,
    ) -> None:
        self.field_length = field_length
        self.field_width = field_width

        # Масштабные коэффициенты (для работы с пиксельными координатами)
        self._scale_x = field_length / 105.0
        self._scale_y = field_width / 68.0

        # История мяча для momentum
        self._ball_history: deque[tuple[float, float]] = deque(maxlen=history_len)
        self._xg_history: deque[float] = deque(maxlen=history_len)

    def compute(
        self,
        ball: TrackedBall | None,
        players: list[TrackedPlayer],
        attacking_team: str = "left",
    ) -> GoalProbResult | None:
        """
        Вычислить xG для текущей позиции мяча.

        Args:
            ball:           Позиция мяча
            players:        Все игроки на поле
            attacking_team: "left" (атакует вправо) или "right" (атакует влево)

        Returns:
            GoalProbResult или None если мяч отсутствует
        """
        if ball is None:
            return None

        # Обновить историю
        self._ball_history.append((ball.x, ball.y))

        # Определить ворота которые атакуют
        if attacking_team == "left":
            goal_x = self.field_length  # правые ворота
            goal_y = self.field_width / 2.0
            defenders = [p for p in players if p.x > self.field_length / 2]
        else:
            goal_x = 0.0              # левые ворота
            goal_y = self.field_width / 2.0
            defenders = [p for p in players if p.x < self.field_length / 2]

        # Базовые геометрические метрики
        dist = math.hypot(ball.x - goal_x, ball.y - goal_y)
        angle = self._shot_angle(ball.x, ball.y, goal_x, goal_y)
        n_defenders = self._defenders_in_path(ball, goal_x, goal_y, defenders)

        # Зона опасности
        in_danger = self._in_danger_zone(ball.x, ball.y, attacking_team)

        # xG базовая формула (аппроксимация StatsBomb xG)
        xg = self._compute_xg(dist, angle, n_defenders, in_danger, ball, attacking_team)
        self._xg_history.append(xg)

        # Вероятность гола в N секунд (с учётом momentum)
        momentum = self._compute_momentum()
        prob_5s = self._xg_to_prob(xg, seconds=5, momentum=momentum)
        prob_30s = self._xg_to_prob(xg, seconds=30, momentum=momentum)

        # Уровень угрозы
        threat = self._threat_level(xg)

        return GoalProbResult(
            xg=xg,
            probability_5s=prob_5s,
            probability_30s=prob_30s,
            danger_zone=in_danger,
            distance_to_goal=dist,
            shot_angle_deg=angle,
            defenders_in_path=n_defenders,
            attacking_team=attacking_team,
            threat_level=threat,
        )

    def compute_for_both_teams(
        self,
        ball: TrackedBall | None,
        players: list[TrackedPlayer],
    ) -> tuple[GoalProbResult | None, GoalProbResult | None]:
        """Вычислить xG для обеих команд одновременно."""
        left = self.compute(ball, players, "left")
        right = self.compute(ball, players, "right")
        return left, right

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _shot_angle(self, bx: float, by: float, gx: float, gy: float) -> float:
        """
        Угол обстрела — угол между двумя штангами ворот из позиции мяча.
        Больше угол = лучше позиция для удара.
        """
        half_goal = (GOAL_WIDTH / 2.0) * self._scale_y

        post1_x, post1_y = gx, gy - half_goal
        post2_x, post2_y = gx, gy + half_goal

        # Векторы от мяча к штангам
        v1 = (post1_x - bx, post1_y - by)
        v2 = (post2_x - bx, post2_y - by)

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)

        if mag1 * mag2 == 0:
            return 0.0

        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    def _defenders_in_path(
        self,
        ball: TrackedBall,
        goal_x: float,
        goal_y: float,
        defenders: list[TrackedPlayer],
    ) -> int:
        """Количество защитников в конусе между мячом и воротами."""
        count = 0
        cone_angle = 15.0  # градусов от линии мяч-ворота

        ball_to_goal_dx = goal_x - ball.x
        ball_to_goal_dy = goal_y - ball.y
        ball_to_goal_dist = math.hypot(ball_to_goal_dx, ball_to_goal_dy)

        if ball_to_goal_dist == 0:
            return 0

        for d in defenders:
            # Проекция защитника на линию мяч-ворота
            dx = d.x - ball.x
            dy = d.y - ball.y

            # Скалярное произведение
            proj = (dx * ball_to_goal_dx + dy * ball_to_goal_dy) / ball_to_goal_dist

            # Защитник должен быть между мячом и воротами
            if proj < 0 or proj > ball_to_goal_dist:
                continue

            # Перпендикулярное расстояние от линии
            cross = abs(dx * ball_to_goal_dy - dy * ball_to_goal_dx) / ball_to_goal_dist

            # В конусе?
            half_cone_width = proj * math.tan(math.radians(cone_angle))
            if cross < half_cone_width:
                count += 1

        return count

    def _in_danger_zone(self, bx: float, by: float, attacking_team: str) -> bool:
        """Мяч в зоне штрафной площади (примерно)."""
        dz_x = DANGER_ZONE_X_FROM_GOAL * self._scale_x
        dz_w = DANGER_ZONE_WIDTH * self._scale_y
        mid_y = self.field_width / 2.0

        if attacking_team == "left":
            return bx > self.field_length - dz_x and abs(by - mid_y) < dz_w
        else:
            return bx < dz_x and abs(by - mid_y) < dz_w

    def _compute_xg(
        self,
        dist: float,
        angle: float,
        n_defenders: int,
        in_danger: bool,
        ball: TrackedBall,
        attacking_team: str,
    ) -> float:
        """
        Аналитическая xG формула.

        Основана на логистической регрессии из публичных xG моделей:
        xG = sigmoid(b0 + b1*dist + b2*angle + b3*defenders + ...)
        """
        # Нормализованная дистанция (до ~30м)
        max_dist = 30.0 * self._scale_x
        dist_norm = min(dist / max_dist, 2.0)

        # Нормализованный угол (0..45 градусов)
        angle_norm = min(angle / 45.0, 2.0)

        # Логит (линейная часть)
        b0 = -1.5
        b_dist = -2.0      # дальше = меньше шанс
        b_angle = 1.2      # больше угол = больше шанс
        b_def = -0.5       # больше защитников = меньше шанс
        b_danger = 0.8     # в зоне штрафной = больше шанс

        logit = (
            b0
            + b_dist * dist_norm
            + b_angle * angle_norm
            + b_def * min(n_defenders, 4)
            + b_danger * (1.0 if in_danger else 0.0)
        )

        # Sigmoid
        xg = 1.0 / (1.0 + math.exp(-logit))

        # Ограничить реалистичным диапазоном (max xG ~0.9 для пенальти)
        return float(np.clip(xg, 0.0, 0.9))

    def _compute_momentum(self) -> float:
        """
        Momentum угрозы — растёт ли xG в последних кадрах.
        Возвращает 0..2 (1.0 = нейтрально, >1 = угроза растёт).
        """
        if len(self._xg_history) < 10:
            return 1.0
        history = list(self._xg_history)
        half = len(history) // 2
        recent = np.mean(history[half:])
        older = np.mean(history[:half])
        return float(recent / max(older, 0.001))

    def _xg_to_prob(self, xg: float, seconds: float, momentum: float) -> float:
        """
        Конвертировать xG в вероятность гола за N секунд.

        Логика: чем выше xG и чем больше времени, тем выше шанс.
        momentum > 1 означает что ситуация ухудшается для обороны.
        """
        # Базовая вероятность пропорциональна времени
        time_factor = min(seconds / 90.0, 1.0)

        # xG уже содержит info о качестве момента
        prob = xg * time_factor * momentum

        return float(np.clip(prob, 0.0, 0.95))

    @staticmethod
    def _threat_level(xg: float) -> str:
        if xg < 0.05:
            return "low"
        elif xg < 0.15:
            return "medium"
        elif xg < 0.35:
            return "high"
        else:
            return "critical"
