"""
FormationDetector — определяет тактическую формацию команды.

Алгоритм:
  1. Разделить игроков на две команды по X-позиции (левая/правая половина)
  2. Убрать вратаря (самый крайний игрок по X)
  3. K-Means кластеризация оставшихся 10 игроков по X-координате
  4. Посчитать количество игроков в каждом кластере → линии формации
  5. Сравнить с шаблонами (4-3-3, 4-4-2, 3-5-2 и т.д.)

Пример:
    detector = FormationDetector()
    result = detector.detect(players, team="left")
    # FormationResult(name="4-3-3", lines=[4,3,3], confidence=0.87)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np
from scipy.cluster.vq import kmeans2

from fie.tracking.pipeline import TrackedPlayer


# ---------------------------------------------------------------------------
# Известные формации
# ---------------------------------------------------------------------------

KNOWN_FORMATIONS: dict[str, list[int]] = {
    "4-3-3":  [4, 3, 3],
    "4-4-2":  [4, 4, 2],
    "4-2-3-1":[4, 2, 3, 1],
    "4-5-1":  [4, 5, 1],
    "3-5-2":  [3, 5, 2],
    "3-4-3":  [3, 4, 3],
    "5-3-2":  [5, 3, 2],
    "5-4-1":  [5, 4, 1],
    "4-1-4-1":[4, 1, 4, 1],
    "4-3-2-1":[4, 3, 2, 1],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FormationLine:
    """Одна линия формации (защита, полузащита, атака)."""
    player_ids: list[int]
    avg_x: float          # средняя позиция линии по длине поля
    avg_y: float          # средняя позиция по ширине
    width: float          # ширина линии (max_y - min_y)
    count: int            # количество игроков


@dataclass
class FormationResult:
    """Результат определения формации."""
    name: str                        # "4-3-3", "4-4-2", ...
    lines: list[int]                 # [4, 3, 3]
    formation_lines: list[FormationLine]  # детали каждой линии
    confidence: float                # 0.0 – 1.0
    team: str                        # "left" / "right"
    goalkeeper_id: int | None        # ID вратаря

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lines": self.lines,
            "confidence": round(self.confidence, 3),
            "team": self.team,
            "goalkeeper_id": self.goalkeeper_id,
            "formation_lines": [
                {
                    "player_ids": fl.player_ids,
                    "avg_x": round(fl.avg_x, 1),
                    "avg_y": round(fl.avg_y, 1),
                    "width": round(fl.width, 1),
                    "count": fl.count,
                }
                for fl in self.formation_lines
            ],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class FormationDetector:
    """
    Определяет формацию команды из позиций игроков.

    Args:
        field_length: Длина поля в метрах (default=105 для реального поля
                      или пиксельные координаты для IdentityCalibration)
        smoothing_frames: Количество кадров для усреднения позиций
                          (сглаживает шум трекинга)
        min_players: Минимум игроков для определения формации
    """

    def __init__(
        self,
        field_length: float = 105.0,
        field_width: float = 68.0,
        smoothing_frames: int = 25,
        min_players: int = 7,
    ) -> None:
        self.field_length = field_length
        self.field_width = field_width
        self.smoothing_frames = smoothing_frames
        self.min_players = min_players

        # История позиций для сглаживания: {player_id: deque[(x, y)]}
        self._pos_history: dict[int, deque] = {}

    def update(self, players: list[TrackedPlayer]) -> None:
        """Обновить историю позиций (вызывать каждый кадр)."""
        for p in players:
            if p.player_id not in self._pos_history:
                self._pos_history[p.player_id] = deque(maxlen=self.smoothing_frames)
            self._pos_history[p.player_id].append((p.x, p.y))

    def detect(
        self,
        players: list[TrackedPlayer],
        team: str = "left",
    ) -> FormationResult | None:
        """
        Определить формацию для одной команды.

        Args:
            players: Список игроков (можно весь список — функция сама
                     разделит по командам через X-позицию)
            team: "left" (x < mid) или "right" (x >= mid)

        Returns:
            FormationResult или None если недостаточно игроков
        """
        mid = self.field_length / 2.0

        # Получить сглаженные позиции
        smoothed: list[tuple[int, float, float]] = []
        for p in players:
            hist = self._pos_history.get(p.player_id)
            if hist:
                xs = [pos[0] for pos in hist]
                ys = [pos[1] for pos in hist]
                sx, sy = float(np.mean(xs)), float(np.mean(ys))
            else:
                sx, sy = p.x, p.y
            smoothed.append((p.player_id, sx, sy))

        # Разделить по команде
        if team == "left":
            team_players = [(pid, x, y) for pid, x, y in smoothed if x < mid]
        else:
            team_players = [(pid, x, y) for pid, x, y in smoothed if x >= mid]

        if len(team_players) < self.min_players:
            return None

        # Найти вратаря — самый крайний игрок по X
        if team == "left":
            gk = min(team_players, key=lambda p: p[1])
        else:
            gk = max(team_players, key=lambda p: p[1])

        gk_id = gk[0]
        outfield = [(pid, x, y) for pid, x, y in team_players if pid != gk_id]

        if len(outfield) < 6:
            return None

        # K-Means кластеризация по X (линии формации)
        n_lines = self._estimate_lines(len(outfield))
        xs = np.array([[x] for _, x, _ in outfield], dtype=np.float32)

        try:
            centroids, labels = kmeans2(xs, n_lines, minit="++", seed=42, iter=20)
        except Exception:
            return None

        # Сортировать линии по X (атака → оборона для правой, оборона → атака для левой)
        order = np.argsort(centroids[:, 0])
        if team == "right":
            order = order[::-1]

        # Построить линии
        formation_lines = []
        line_counts = []
        for rank, cluster_idx in enumerate(order):
            cluster_players = [
                (outfield[i][0], outfield[i][1], outfield[i][2])
                for i in range(len(outfield))
                if labels[i] == cluster_idx
            ]
            if not cluster_players:
                continue
            pids = [cp[0] for cp in cluster_players]
            cxs = [cp[1] for cp in cluster_players]
            cys = [cp[2] for cp in cluster_players]
            formation_lines.append(FormationLine(
                player_ids=pids,
                avg_x=float(np.mean(cxs)),
                avg_y=float(np.mean(cys)),
                width=float(max(cys) - min(cys)) if len(cys) > 1 else 0.0,
                count=len(pids),
            ))
            line_counts.append(len(pids))

        # Сопоставить с известными формациями
        name, confidence = self._match_formation(line_counts)

        return FormationResult(
            name=name,
            lines=line_counts,
            formation_lines=formation_lines,
            confidence=confidence,
            team=team,
            goalkeeper_id=gk_id,
        )

    def detect_both_teams(
        self,
        players: list[TrackedPlayer],
    ) -> tuple[FormationResult | None, FormationResult | None]:
        """Определить формацию обеих команд за один вызов."""
        left = self.detect(players, team="left")
        right = self.detect(players, team="right")
        return left, right

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_lines(self, n_outfield: int) -> int:
        """Оценить количество линий формации по числу полевых игроков."""
        if n_outfield <= 8:
            return 3
        elif n_outfield <= 10:
            return 3
        else:
            return 4

    def _match_formation(self, line_counts: list[int]) -> tuple[str, float]:
        """
        Сопоставить список линий с известными формациями.

        Returns:
            (name, confidence) — название формации и уверенность 0..1
        """
        if not line_counts:
            return "unknown", 0.0

        best_name = "unknown"
        best_score = 0.0

        for name, template in KNOWN_FORMATIONS.items():
            score = self._line_similarity(line_counts, template)
            if score > best_score:
                best_score = score
                best_name = name

        # Если ни одна не подошла — сформировать строку из counts
        if best_score < 0.5:
            best_name = "-".join(str(c) for c in line_counts)

        return best_name, best_score

    @staticmethod
    def _line_similarity(a: list[int], b: list[int]) -> float:
        """
        Мера схожести двух формаций.
        1.0 = полное совпадение, 0.0 = полностью разные.
        """
        if len(a) != len(b):
            # Разное кол-во линий — штраф, но не 0
            min_len = min(len(a), len(b))
            a_trim = sorted(a, reverse=True)[:min_len]
            b_trim = sorted(b, reverse=True)[:min_len]
            penalty = 0.8
        else:
            a_trim = a
            b_trim = b
            penalty = 1.0

        if not a_trim:
            return 0.0

        total_diff = sum(abs(x - y) for x, y in zip(a_trim, b_trim))
        max_diff = sum(max(x, y) for x, y in zip(a_trim, b_trim))
        similarity = 1.0 - (total_diff / max(max_diff, 1))
        return similarity * penalty
