"""
TacticalPipeline — объединяет Formation, Pressing и Compactness.

Принимает поток TrackingFrame, на каждом кадре:
  1. Обновляет историю позиций (FormationDetector)
  2. Определяет формацию обеих команд (каждые N кадров)
  3. Анализирует прессинг (каждый кадр)
  4. Вычисляет компактность (каждый кадр)

Использование:
    pipeline = TacticalPipeline()
    for frame in tracking_frames:
        result = pipeline.process_frame(frame)
        if result:
            print(result.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fie.tactical.compactness import CompactnessAnalyzer, CompactnessSnapshot
from fie.tactical.formation import FormationDetector, FormationResult
from fie.tactical.pressing import PressingAnalyzer, PressingSnapshot, PressingStats
from fie.tracking.pipeline import TrackingFrame


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TacticalFrame:
    """Тактический анализ одного кадра."""
    frame_idx: int
    timestamp: float

    # Формация (обновляется каждые formation_interval кадров)
    formation_left: FormationResult | None
    formation_right: FormationResult | None

    # Прессинг
    pressing_left: PressingSnapshot | None
    pressing_right: PressingSnapshot | None

    # Компактность
    compactness: CompactnessSnapshot | None

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "formation_left": self.formation_left.to_dict() if self.formation_left else None,
            "formation_right": self.formation_right.to_dict() if self.formation_right else None,
            "pressing_left": self.pressing_left.to_dict() if self.pressing_left else None,
            "pressing_right": self.pressing_right.to_dict() if self.pressing_right else None,
            "compactness": self.compactness.to_dict() if self.compactness else None,
        }


@dataclass
class TacticalSummary:
    """Сводный тактический отчёт за матч / отрезок."""
    total_frames: int
    duration_seconds: float

    # Доминирующая формация (самая частая)
    dominant_formation_left: str | None
    dominant_formation_right: str | None

    # Статистика прессинга
    pressing_stats_left: dict | None
    pressing_stats_right: dict | None

    # Средняя компактность
    avg_compactness_left: float | None
    avg_compactness_right: float | None
    avg_inter_team_distance: float | None

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 1),
            "dominant_formation_left": self.dominant_formation_left,
            "dominant_formation_right": self.dominant_formation_right,
            "pressing_stats_left": self.pressing_stats_left,
            "pressing_stats_right": self.pressing_stats_right,
            "avg_compactness_left": (
                round(self.avg_compactness_left, 3) if self.avg_compactness_left else None
            ),
            "avg_compactness_right": (
                round(self.avg_compactness_right, 3) if self.avg_compactness_right else None
            ),
            "avg_inter_team_distance": (
                round(self.avg_inter_team_distance, 1) if self.avg_inter_team_distance else None
            ),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TacticalPipeline:
    """
    Полный тактический анализ матча.

    Args:
        field_length:         Длина поля (105м или пиксели)
        field_width:          Ширина поля (68м или пиксели)
        formation_interval:   Каждые N кадров пересчитывать формацию (default=25)
        press_radius:         Радиус прессинга вокруг мяча
    """

    def __init__(
        self,
        field_length: float = 105.0,
        field_width: float = 68.0,
        formation_interval: int = 25,
        press_radius: float = 5.0,
    ) -> None:
        self.field_length = field_length
        self.field_width = field_width
        self.formation_interval = formation_interval

        self._formation = FormationDetector(
            field_length=field_length,
            field_width=field_width,
        )
        self._pressing = PressingAnalyzer(
            field_length=field_length,
            press_radius=press_radius,
        )
        self._compactness = CompactnessAnalyzer(
            field_length=field_length,
            field_width=field_width,
        )

        # Кэш последней формации (обновляется редко)
        self._last_formation_left: FormationResult | None = None
        self._last_formation_right: FormationResult | None = None
        self._frame_count = 0

        # Для сводки
        self._formation_left_counts: dict[str, int] = {}
        self._formation_right_counts: dict[str, int] = {}
        self._compactness_left_sum = 0.0
        self._compactness_right_sum = 0.0
        self._inter_team_dist_sum = 0.0
        self._compactness_frames = 0

    def process_frame(self, frame: TrackingFrame) -> TacticalFrame:
        """
        Обработать один трекинг-кадр.

        Returns:
            TacticalFrame с тактическими метриками
        """
        self._frame_count += 1
        players = frame.players
        ball = frame.ball

        # 1. Обновить историю позиций
        self._formation.update(players)

        # 2. Пересчитать формацию каждые N кадров
        if self._frame_count % self.formation_interval == 0 or self._frame_count == 1:
            left_f, right_f = self._formation.detect_both_teams(players)
            if left_f:
                self._last_formation_left = left_f
                name = left_f.name
                self._formation_left_counts[name] = self._formation_left_counts.get(name, 0) + 1
            if right_f:
                self._last_formation_right = right_f
                name = right_f.name
                self._formation_right_counts[name] = self._formation_right_counts.get(name, 0) + 1

        # 3. Прессинг (каждый кадр)
        pressing_left, pressing_right = self._pressing.analyze_frame(players, ball)

        # 4. Компактность (каждый кадр)
        compactness = self._compactness.analyze(players)
        if compactness.left:
            self._compactness_left_sum += compactness.left.compactness
            self._compactness_frames += 1
        if compactness.right:
            self._compactness_right_sum += compactness.right.compactness
        if compactness.inter_team_distance is not None:
            self._inter_team_dist_sum += compactness.inter_team_distance

        return TacticalFrame(
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            formation_left=self._last_formation_left,
            formation_right=self._last_formation_right,
            pressing_left=pressing_left,
            pressing_right=pressing_right,
            compactness=compactness,
        )

    def process_frames(self, frames: list[TrackingFrame]) -> list[TacticalFrame]:
        """Обработать список кадров, вернуть список TacticalFrame."""
        return [self.process_frame(f) for f in frames]

    def get_summary(self, fps: float = 25.0) -> TacticalSummary:
        """Сводный отчёт за весь обработанный период."""
        n = self._frame_count

        # Доминирующая формация = самая часто встречавшаяся
        dom_left = (
            max(self._formation_left_counts, key=self._formation_left_counts.get)
            if self._formation_left_counts else None
        )
        dom_right = (
            max(self._formation_right_counts, key=self._formation_right_counts.get)
            if self._formation_right_counts else None
        )

        # Статистика прессинга
        press_left_stats, press_right_stats = self._pressing.get_stats()

        # Средняя компактность
        cf = max(self._compactness_frames, 1)
        avg_compact_left = self._compactness_left_sum / cf if self._compactness_frames else None
        avg_compact_right = self._compactness_right_sum / cf if self._compactness_frames else None
        avg_inter = self._inter_team_dist_sum / cf if self._compactness_frames else None

        return TacticalSummary(
            total_frames=n,
            duration_seconds=n / fps,
            dominant_formation_left=dom_left,
            dominant_formation_right=dom_right,
            pressing_stats_left=press_left_stats.to_dict() if press_left_stats else None,
            pressing_stats_right=press_right_stats.to_dict() if press_right_stats else None,
            avg_compactness_left=avg_compact_left,
            avg_compactness_right=avg_compact_right,
            avg_inter_team_distance=avg_inter,
        )
