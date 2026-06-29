"""
Team DNA — характер игры команды.

Анализирует и накапливает стиль игры команды за несколько матчей:

  pressing_style     : high_press / mid_block / deep_block
  tempo              : fast / medium / slow (темп передач и движения)
  territory          : attacking / balanced / defensive (% времени с мячом в чужой трети)
  defensive_line     : high / medium / low (средняя позиция защитной линии)
  attack_width       : wide / central / mixed (ширина атаки)
  aggression         : высокая интенсивность единоборств и прессинга
  compactness        : компактность блока
  directness         : прямолинейность атак (вертикальность vs. контроль мяча)

DNA-профиль — это вектор из 8 числовых значений [0..1].
Похожие команды имеют маленькое евклидово расстояние между ДНК-векторами.

Хранение: JSON в data/team_dna/{team_id}.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import numpy as np

from fie.tracking.pipeline import TrackingFrame
from fie.tactical.pressing import PressingAnalyzer
from fie.tactical.compactness import CompactnessAnalyzer


# ---------------------------------------------------------------------------
# DNA Profile
# ---------------------------------------------------------------------------

@dataclass
class TeamDNAVector:
    """8-мерный вектор ДНК команды (все значения 0..1)."""
    pressing_intensity: float   # интенсивность прессинга
    pressing_line: float        # высота линии прессинга
    tempo: float                # темп игры
    territory: float            # территориальное доминирование
    defensive_line: float       # высота оборонительной линии
    attack_width: float         # ширина атаки
    aggression: float           # агрессивность
    compactness: float          # компактность блока

    def to_list(self) -> list[float]:
        return [
            self.pressing_intensity,
            self.pressing_line,
            self.tempo,
            self.territory,
            self.defensive_line,
            self.attack_width,
            self.aggression,
            self.compactness,
        ]

    def distance(self, other: "TeamDNAVector") -> float:
        """Евклидово расстояние между двумя ДНК-векторами."""
        a = np.array(self.to_list())
        b = np.array(other.to_list())
        return float(np.linalg.norm(a - b))

    def similarity(self, other: "TeamDNAVector") -> float:
        """Сходство 0..1 (1 = идентичный стиль)."""
        max_dist = np.sqrt(8)  # максимальное расстояние при всех 0 vs 1
        return max(0.0, 1.0 - self.distance(other) / max_dist)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TeamDNAProfile:
    """Полный ДНК-профиль команды."""
    team_id: str
    team_name: str
    total_matches: int

    # Средний ДНК-вектор
    dna: TeamDNAVector

    # Текстовые ярлыки
    pressing_style: str        # "high_press" / "mid_block" / "deep_block"
    tempo_label: str           # "fast" / "medium" / "slow"
    territory_label: str       # "attacking" / "balanced" / "defensive"
    defensive_line_label: str  # "high" / "medium" / "low"
    attack_style: str          # "wide" / "central" / "mixed"

    # Краткое описание стиля
    style_summary: str

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "total_matches": self.total_matches,
            "dna": self.dna.to_dict(),
            "pressing_style": self.pressing_style,
            "tempo_label": self.tempo_label,
            "territory_label": self.territory_label,
            "defensive_line_label": self.defensive_line_label,
            "attack_style": self.attack_style,
            "style_summary": self.style_summary,
        }


# ---------------------------------------------------------------------------
# Per-match DNA extractor
# ---------------------------------------------------------------------------

@dataclass
class _MatchDNAAccumulator:
    """Накапливает ДНК за один матч."""
    # Прессинг
    pressing_vals: list[float] = field(default_factory=list)
    pressing_line_vals: list[float] = field(default_factory=list)
    # Территория
    territory_vals: list[float] = field(default_factory=list)
    # Оборонительная линия
    def_line_vals: list[float] = field(default_factory=list)
    # Ширина атаки (разброс Y)
    attack_y_vals: list[float] = field(default_factory=list)
    # Скорость
    speed_vals: list[float] = field(default_factory=list)
    # Компактность
    compactness_vals: list[float] = field(default_factory=list)
    # Кол-во спринтов (агрессивность)
    sprint_frames: int = 0
    total_frames: int = 0


def _extract_match_dna(
    frames: list[TrackingFrame],
    team: str,
    field_length: float = 105.0,
    field_width: float = 68.0,
) -> TeamDNAVector:
    """Извлечь ДНК-вектор команды из кадров одного матча."""
    acc = _MatchDNAAccumulator()
    pressing_analyzer = PressingAnalyzer(field_length=field_length, field_width=field_width)
    compactness_analyzer = CompactnessAnalyzer(field_length=field_length, field_width=field_width)

    for frame in frames:
        players = frame.players
        ball = frame.ball
        if not players:
            continue

        acc.total_frames += 1

        # Определяем "нашу" и "чужую" команды по X-позиции
        xs = [p.x for p in players]
        mid_x = field_length / 2

        # Игроки в "левой" и "правой" группах (упрощённо по X)
        left = [p for p in players if p.x < mid_x]
        right = [p for p in players if p.x >= mid_x]
        our_players = left if team == "left" else right

        if not our_players:
            continue

        # Прессинг
        if ball:
            press_snap = pressing_analyzer.analyze_frame(players, ball)
            if press_snap:
                acc.pressing_vals.append(press_snap.pressing_intensity)
                acc.pressing_line_vals.append(press_snap.pressing_line / field_length)

        # Компактность
        comp_snap = compactness_analyzer.analyze_frame(players)
        if comp_snap:
            acc.compactness_vals.append(comp_snap.home_compactness)

        # Территория (% времени мяч в чужой трети)
        if ball:
            if team == "left":
                in_attack = ball.x > field_length * 2 / 3
            else:
                in_attack = ball.x < field_length / 3
            acc.territory_vals.append(1.0 if in_attack else 0.0)

        # Оборонительная линия (средний X защитников)
        our_xs = [p.x for p in our_players]
        def_line = min(our_xs) / field_length if team == "left" else 1.0 - max(our_xs) / field_length
        acc.def_line_vals.append(max(0.0, min(1.0, def_line)))

        # Ширина атаки (разброс Y)
        our_ys = [p.y for p in our_players]
        if len(our_ys) >= 2:
            width = (max(our_ys) - min(our_ys)) / field_width
            acc.attack_y_vals.append(min(width, 1.0))

        # Темп (средняя скорость)
        speeds = [p.speed for p in our_players if p.speed > 0]
        if speeds:
            acc.speed_vals.append(float(np.mean(speeds)))

        # Агрессивность (доля игроков в спринте)
        sprints = sum(1 for p in our_players if p.speed >= 25.0)
        if sprints > 0:
            acc.sprint_frames += 1

    def _avg(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    pressing_intensity = _avg(acc.pressing_vals)
    pressing_line = _avg(acc.pressing_line_vals)
    tempo_raw = _avg(acc.speed_vals)
    # Нормализуем скорость: 0 км/ч = 0, 15+ км/ч = 1
    tempo = min(tempo_raw / 15.0, 1.0)
    territory = _avg(acc.territory_vals)
    defensive_line = 1.0 - _avg(acc.def_line_vals)  # инвертируем: высокая = 1
    attack_width = _avg(acc.attack_y_vals)
    aggression = acc.sprint_frames / max(acc.total_frames, 1)
    compactness = _avg(acc.compactness_vals)

    return TeamDNAVector(
        pressing_intensity=round(pressing_intensity, 3),
        pressing_line=round(pressing_line, 3),
        tempo=round(tempo, 3),
        territory=round(territory, 3),
        defensive_line=round(defensive_line, 3),
        attack_width=round(attack_width, 3),
        aggression=round(min(aggression * 5, 1.0), 3),  # ×5 для нормализации
        compactness=round(compactness, 3),
    )


# ---------------------------------------------------------------------------
# TeamDNA class (хранилище + профиль)
# ---------------------------------------------------------------------------

class TeamDNA:
    """
    Накапливает ДНК команды за несколько матчей.

    Args:
        team_id:    Уникальный ID команды
        team_name:  Название для отображения
        save_dir:   Директория для JSON файлов
    """

    def __init__(
        self,
        team_id: str,
        team_name: str = "",
        save_dir: str | Path = "data/team_dna",
    ) -> None:
        self.team_id = team_id
        self.team_name = team_name or team_id
        self.save_dir = Path(save_dir)
        self._match_dnas: list[TeamDNAVector] = []

    @classmethod
    def load(cls, path: str | Path) -> "TeamDNA":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(
            team_id=data["team_id"],
            team_name=data.get("team_name", data["team_id"]),
            save_dir=path.parent,
        )
        for m in data.get("matches", []):
            obj._match_dnas.append(TeamDNAVector(**m))
        return obj

    @classmethod
    def load_or_create(
        cls,
        team_id: str,
        team_name: str = "",
        save_dir: str | Path = "data/team_dna",
    ) -> "TeamDNA":
        path = Path(save_dir) / f"{team_id}.json"
        if path.exists():
            return cls.load(path)
        return cls(team_id=team_id, team_name=team_name, save_dir=save_dir)

    def add_match(
        self,
        frames: list[TrackingFrame],
        team: str = "left",
        field_length: float = 105.0,
        field_width: float = 68.0,
    ) -> TeamDNAVector:
        """Добавить матч и обновить ДНК."""
        dna = _extract_match_dna(frames, team, field_length, field_width)
        self._match_dnas.append(dna)
        return dna

    def add_match_dna(self, dna: TeamDNAVector) -> None:
        """Добавить уже извлечённый ДНК-вектор."""
        self._match_dnas.append(dna)

    def save(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{self.team_id}.json"
        data = {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "updated_at": datetime.now().isoformat(),
            "matches": [asdict(d) for d in self._match_dnas],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def get_profile(self) -> TeamDNAProfile | None:
        if not self._match_dnas:
            return None

        # Усредняем все матчи
        keys = ["pressing_intensity", "pressing_line", "tempo", "territory",
                "defensive_line", "attack_width", "aggression", "compactness"]
        avg = {}
        for k in keys:
            avg[k] = float(np.mean([getattr(d, k) for d in self._match_dnas]))

        dna = TeamDNAVector(**{k: round(v, 3) for k, v in avg.items()})

        # Ярлыки
        if dna.pressing_intensity > 0.6:
            pressing_style = "high_press"
        elif dna.pressing_intensity > 0.3:
            pressing_style = "mid_block"
        else:
            pressing_style = "deep_block"

        tempo_label = "fast" if dna.tempo > 0.6 else ("medium" if dna.tempo > 0.3 else "slow")
        territory_label = "attacking" if dna.territory > 0.55 else ("defensive" if dna.territory < 0.4 else "balanced")
        def_line_label = "high" if dna.defensive_line > 0.6 else ("medium" if dna.defensive_line > 0.35 else "low")
        attack_style = "wide" if dna.attack_width > 0.6 else ("central" if dna.attack_width < 0.4 else "mixed")

        style_summary = (
            f"{pressing_style.replace('_', ' ').title()} team with "
            f"{tempo_label} tempo, {territory_label} territorial approach, "
            f"{attack_style} attack width and "
            f"{'high' if dna.aggression > 0.5 else 'moderate'} aggression."
        )

        return TeamDNAProfile(
            team_id=self.team_id,
            team_name=self.team_name,
            total_matches=len(self._match_dnas),
            dna=dna,
            pressing_style=pressing_style,
            tempo_label=tempo_label,
            territory_label=territory_label,
            defensive_line_label=def_line_label,
            attack_style=attack_style,
            style_summary=style_summary,
        )

    def compare(self, other: "TeamDNA") -> dict:
        """Сравнить ДНК двух команд."""
        p1 = self.get_profile()
        p2 = other.get_profile()
        if not p1 or not p2:
            return {"error": "Insufficient data"}
        sim = p1.dna.similarity(p2.dna)
        diff = {
            k: round(getattr(p1.dna, k) - getattr(p2.dna, k), 3)
            for k in ["pressing_intensity", "tempo", "territory", "aggression", "compactness"]
        }
        return {
            "similarity": round(sim, 3),
            "team1": p1.to_dict(),
            "team2": p2.to_dict(),
            "differences": diff,
        }
