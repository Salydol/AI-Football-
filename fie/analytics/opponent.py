"""
Opponent Weakness Scanner — анализ слабостей соперника.

Принцип работы:
  1. Загружаем данные нескольких матчей соперника (TeamDNA, статистика, ошибки)
  2. Находим паттерны: слабые зоны, физический спад, уязвимые игроки
  3. Генерируем план атаки и рекомендации по игровой стратегии

Пример использования:
    scanner = OpponentScanner("Real Madrid")
    scanner.add_match(OpponentMatch(...))
    scanner.add_match(OpponentMatch(...))
    report = scanner.analyze()
    print(report.attack_recommendations)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Literal


# ---------------------------------------------------------------------------
# Input data structures
# ---------------------------------------------------------------------------

@dataclass
class ZoneStats:
    """Статистика по зоне поля (left / center / right × def / mid / att)."""
    zone: str          # e.g. "left_def", "center_mid", "right_att"
    losses: int = 0    # потери мяча в зоне
    mistakes: int = 0  # тактические ошибки
    goals_conceded: int = 0
    xg_conceded: float = 0.0
    pressing_failures: int = 0


@dataclass
class OpponentMatch:
    """Данные одного матча соперника."""
    match_id: str
    date: str
    opponent_name: str          # кто играл против нашего будущего соперника

    # Физика
    distance_km: float = 0.0
    sprint_count: int = 0
    max_speed_kmh: float = 0.0
    high_accel_count: int = 0

    # Тактика
    pressing_intensity: float = 0.5   # 0..1
    pressing_line: float = 0.5        # 0..1 (высокий/низкий блок)
    compactness: float = 0.5
    territory: float = 0.5            # % владения территорией

    # Результат
    goals_scored: int = 0
    goals_conceded: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0

    # Ошибки и слабые зоны
    defensive_mistakes: list[dict] = field(default_factory=list)
    zone_stats: list[ZoneStats] = field(default_factory=list)

    # Физический спад (по минутам)
    speed_by_period: dict[str, float] = field(default_factory=dict)
    # {"0-15": 8.2, "15-30": 7.9, "30-45": 7.4, "45-60": 7.1, "60-75": 6.8, "75-90": 6.2}

    # Слабые игроки (player_id → weakness description)
    weak_players: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output structures
# ---------------------------------------------------------------------------

@dataclass
class WeakZone:
    zone: str
    weakness_score: float    # 0..1 (1 = максимальная слабость)
    avg_mistakes: float
    avg_losses: float
    avg_xg_conceded: float
    description: str


@dataclass
class FatigueWindow:
    period: str              # "60-75", "75-90"
    avg_speed_drop_pct: float
    description: str


@dataclass
class TacticalVulnerability:
    type: str                # "high_press_susceptible", "slow_transition", etc.
    severity: Literal["low", "medium", "high"]
    description: str
    recommendation: str


@dataclass
class OpponentWeaknessReport:
    opponent_name: str
    matches_analyzed: int
    generated_at: str

    # Слабые зоны
    weak_zones: list[WeakZone]

    # Физический спад
    fatigue_windows: list[FatigueWindow]

    # Тактические уязвимости
    tactical_vulnerabilities: list[TacticalVulnerability]

    # Слабые игроки
    weak_players: dict[int, str]

    # Готовые рекомендации по атаке
    attack_recommendations: list[str]

    # Общий рейтинг уязвимости 0..100
    overall_vulnerability_score: float

    def to_dict(self) -> dict:
        return {
            "opponent_name": self.opponent_name,
            "matches_analyzed": self.matches_analyzed,
            "generated_at": self.generated_at,
            "overall_vulnerability_score": round(self.overall_vulnerability_score, 1),
            "weak_zones": [asdict(z) for z in self.weak_zones],
            "fatigue_windows": [asdict(fw) for fw in self.fatigue_windows],
            "tactical_vulnerabilities": [asdict(v) for v in self.tactical_vulnerabilities],
            "weak_players": {str(k): v for k, v in self.weak_players.items()},
            "attack_recommendations": self.attack_recommendations,
        }

    def save(self, save_dir: str | Path = "data/opponent_reports") -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.opponent_name.replace(" ", "_").lower()
        path = save_dir / f"{safe_name}_weakness.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OpponentWeaknessReport":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            opponent_name=d["opponent_name"],
            matches_analyzed=d["matches_analyzed"],
            generated_at=d["generated_at"],
            overall_vulnerability_score=d["overall_vulnerability_score"],
            weak_zones=[WeakZone(**z) for z in d.get("weak_zones", [])],
            fatigue_windows=[FatigueWindow(**fw) for fw in d.get("fatigue_windows", [])],
            tactical_vulnerabilities=[TacticalVulnerability(**v) for v in d.get("tactical_vulnerabilities", [])],
            weak_players={int(k): v for k, v in d.get("weak_players", {}).items()},
            attack_recommendations=d.get("attack_recommendations", []),
        )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class OpponentScanner:
    """
    Анализирует слабости соперника по нескольким матчам.

    Usage:
        scanner = OpponentScanner("FC Barcelona")
        for match_data in matches:
            scanner.add_match(match_data)
        report = scanner.analyze()
    """

    def __init__(self, opponent_name: str) -> None:
        self.opponent_name = opponent_name
        self._matches: list[OpponentMatch] = []

    def add_match(self, match: OpponentMatch) -> None:
        self._matches.append(match)

    @property
    def match_count(self) -> int:
        return len(self._matches)

    def analyze(self) -> OpponentWeaknessReport:
        """Запустить полный анализ слабостей."""
        from datetime import datetime
        if not self._matches:
            raise ValueError("No matches added. Use add_match() first.")

        weak_zones      = self._analyze_zones()
        fatigue_windows = self._analyze_fatigue()
        vulnerabilities = self._analyze_tactics()
        weak_players    = self._aggregate_weak_players()
        recommendations = self._build_recommendations(weak_zones, fatigue_windows, vulnerabilities)
        vuln_score      = self._vulnerability_score(weak_zones, vulnerabilities)

        return OpponentWeaknessReport(
            opponent_name=self.opponent_name,
            matches_analyzed=len(self._matches),
            generated_at=datetime.now().isoformat(),
            weak_zones=weak_zones,
            fatigue_windows=fatigue_windows,
            tactical_vulnerabilities=vulnerabilities,
            weak_players=weak_players,
            attack_recommendations=recommendations,
            overall_vulnerability_score=vuln_score,
        )

    # ── Zone analysis ──────────────────────────────────────────────────────

    def _analyze_zones(self) -> list[WeakZone]:
        """Агрегируем статистику по зонам по всем матчам."""
        zone_data: dict[str, list[ZoneStats]] = {}
        for match in self._matches:
            for zs in match.zone_stats:
                zone_data.setdefault(zs.zone, []).append(zs)

        results: list[WeakZone] = []
        for zone, stats_list in zone_data.items():
            avg_mistakes  = mean(s.mistakes for s in stats_list)
            avg_losses    = mean(s.losses for s in stats_list)
            avg_xg        = mean(s.xg_conceded for s in stats_list)
            avg_pf        = mean(s.pressing_failures for s in stats_list)

            # weakness_score: взвешенная сумма
            score = min(1.0, (avg_mistakes * 0.3 + avg_losses * 0.1 +
                               avg_xg * 0.4 + avg_pf * 0.2) / 5.0)

            if score > 0.15:  # только значимые слабости
                desc = _zone_description(zone, avg_mistakes, avg_losses, avg_xg)
                results.append(WeakZone(
                    zone=zone,
                    weakness_score=round(score, 3),
                    avg_mistakes=round(avg_mistakes, 2),
                    avg_losses=round(avg_losses, 2),
                    avg_xg_conceded=round(avg_xg, 3),
                    description=desc,
                ))

        results.sort(key=lambda z: z.weakness_score, reverse=True)
        return results[:5]

    # ── Fatigue analysis ───────────────────────────────────────────────────

    def _analyze_fatigue(self) -> list[FatigueWindow]:
        """Находим периоды физического спада."""
        periods = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]
        period_speeds: dict[str, list[float]] = {p: [] for p in periods}

        for match in self._matches:
            for period, speed in match.speed_by_period.items():
                if period in period_speeds:
                    period_speeds[period].append(speed)

        # Рассчитываем средние и находим спад
        avgs: dict[str, float] = {}
        for p in periods:
            if period_speeds[p]:
                avgs[p] = mean(period_speeds[p])

        if len(avgs) < 2:
            return []

        baseline = avgs.get("0-15") or next(iter(avgs.values()))
        results: list[FatigueWindow] = []
        for p in periods[2:]:  # начиная с 30-й минуты
            if p not in avgs:
                continue
            drop_pct = (baseline - avgs[p]) / baseline * 100 if baseline > 0 else 0
            if drop_pct >= 5.0:  # значимый спад (≥5%)
                results.append(FatigueWindow(
                    period=p,
                    avg_speed_drop_pct=round(drop_pct, 1),
                    description=(
                        f"В период {p} мин средняя скорость падает на {drop_pct:.1f}%. "
                        f"Соперник уязвим — увеличьте темп и давление."
                    ),
                ))

        return results

    # ── Tactical analysis ──────────────────────────────────────────────────

    def _analyze_tactics(self) -> list[TacticalVulnerability]:
        """Находим тактические паттерны уязвимости."""
        vulnerabilities: list[TacticalVulnerability] = []

        avg_pressing   = mean(m.pressing_intensity for m in self._matches)
        avg_compactness = mean(m.compactness for m in self._matches)
        avg_xg_against = mean(m.xg_against for m in self._matches)
        avg_conceded   = mean(m.goals_conceded for m in self._matches)

        # Высокий прессинг → уязвим к быстрым переходам
        if avg_pressing > 0.65:
            vulnerabilities.append(TacticalVulnerability(
                type="high_press_counter_vulnerable",
                severity="high",
                description=(
                    f"Соперник играет с высоким прессингом (intensity={avg_pressing:.2f}). "
                    "После потери мяча остаётся много пространства за спиной защитников."
                ),
                recommendation=(
                    "Используйте быстрые контратаки через 1-2 передачи. "
                    "Нападающие должны играть на опережение за линией обороны."
                ),
            ))

        # Слабая компактность → уязвим к комбинационной игре
        if avg_compactness < 0.4:
            vulnerabilities.append(TacticalVulnerability(
                type="poor_compactness",
                severity="high",
                description=(
                    f"Низкая компактность команды (avg={avg_compactness:.2f}). "
                    "Между линиями образуются большие разрывы."
                ),
                recommendation=(
                    "Атакуйте через пространство между линиями. "
                    "Разыгрывайте комбинации с подключением полузащиты в разрывы."
                ),
            ))

        # Много пропускают (xG)
        if avg_xg_against > 1.5:
            severity = "high" if avg_xg_against > 2.0 else "medium"
            vulnerabilities.append(TacticalVulnerability(
                type="defensive_fragility",
                severity=severity,
                description=(
                    f"Соперник регулярно допускает высокий xG против ({avg_xg_against:.2f}/матч). "
                    "Оборона создаёт опасные моменты для соперника."
                ),
                recommendation=(
                    "Играйте активно в атаке с первых минут. "
                    "Создавайте давление и не давайте сопернику организовать оборону."
                ),
            ))

        # Низкий прессинг → уязвим к контролю мяча
        if avg_pressing < 0.35:
            vulnerabilities.append(TacticalVulnerability(
                type="passive_defense",
                severity="medium",
                description=(
                    f"Соперник играет пассивно в обороне (pressing={avg_pressing:.2f}). "
                    "Не создаёт давления на владеющего мячом."
                ),
                recommendation=(
                    "Контролируйте мяч и терпеливо ищите бреши. "
                    "Переключайтесь быстро при потере — соперник не успевает прессинговать."
                ),
            ))

        # Много пропущенных голов в среднем
        if avg_conceded > 1.5:
            vulnerabilities.append(TacticalVulnerability(
                type="high_goals_conceded",
                severity="medium",
                description=(
                    f"Соперник пропускает в среднем {avg_conceded:.1f} гола за матч. "
                    "Стандартные обороны не справляются с нагрузкой."
                ),
                recommendation=(
                    "Атакуйте стандартными положениями и подачами. "
                    "Численное преимущество в штрафной площадке соперника — приоритет."
                ),
            ))

        return vulnerabilities

    # ── Weak players aggregation ───────────────────────────────────────────

    def _aggregate_weak_players(self) -> dict[int, str]:
        """Объединяем данные о слабых игроках из всех матчей."""
        player_mentions: dict[int, list[str]] = {}
        for match in self._matches:
            for pid, desc in match.weak_players.items():
                player_mentions.setdefault(pid, []).append(desc)

        # Оставляем тех, кто упомянут в 2+ матчах
        result: dict[int, str] = {}
        for pid, descs in player_mentions.items():
            if len(descs) >= max(1, len(self._matches) // 2):
                result[pid] = descs[0]  # берём первое описание
        return result

    # ── Recommendations ────────────────────────────────────────────────────

    def _build_recommendations(
        self,
        zones: list[WeakZone],
        fatigue: list[FatigueWindow],
        vulns: list[TacticalVulnerability],
    ) -> list[str]:
        recs: list[str] = []

        # Топ слабая зона
        if zones:
            top = zones[0]
            recs.append(
                f"Атакуйте зону '{top.zone}' — там {top.avg_mistakes:.0f} ошибок/матч "
                f"и xG против {top.avg_xg_conceded:.2f}."
            )

        # Усиление давления в период спада
        if fatigue:
            worst = max(fatigue, key=lambda fw: fw.avg_speed_drop_pct)
            recs.append(
                f"Усиливайте давление в {worst.period} мин — скорость соперника "
                f"падает на {worst.avg_speed_drop_pct:.1f}%."
            )

        # Тактические рекомендации из уязвимостей
        for v in vulns:
            if v.recommendation and v.severity in ("high", "medium"):
                recs.append(v.recommendation)

        # Слабые игроки
        wp = self._aggregate_weak_players()
        if wp:
            pids = ", ".join(f"#{pid}" for pid in list(wp.keys())[:3])
            recs.append(f"Акцентируйте давление на игроков {pids} — выявленные уязвимости.")

        return recs[:8]  # топ-8 рекомендаций

    # ── Vulnerability score ────────────────────────────────────────────────

    def _vulnerability_score(
        self,
        zones: list[WeakZone],
        vulns: list[TacticalVulnerability],
    ) -> float:
        score = 0.0
        if zones:
            score += mean(z.weakness_score for z in zones) * 40
        severity_map = {"low": 5, "medium": 10, "high": 18}
        for v in vulns:
            score += severity_map.get(v.severity, 5)
        return min(100.0, round(score, 1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zone_description(zone: str, mistakes: float, losses: float, xg: float) -> str:
    zone_names = {
        "left_def":    "левый фланг обороны",
        "center_def":  "центр обороны",
        "right_def":   "правый фланг обороны",
        "left_mid":    "левый фланг полузащиты",
        "center_mid":  "центр полузащиты",
        "right_mid":   "правый фланг полузащиты",
        "left_att":    "левый фланг атаки",
        "center_att":  "центр атаки",
        "right_att":   "правый фланг атаки",
    }
    name = zone_names.get(zone, zone)
    return (
        f"Зона '{name}': {mistakes:.1f} ошибок/матч, "
        f"{losses:.1f} потерь/матч, xG допущено {xg:.2f}."
    )
