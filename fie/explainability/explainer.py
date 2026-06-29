"""
Explainable AI — объяснения для предсказаний и аналитики FIE.

Принцип: каждое предсказание системы сопровождается понятным объяснением
на русском или английском языке — «почему система так решила».

Примеры:
  • Вероятность гола 38% → «Потому что: потеря компактности, открытый правый фланг,
    численное преимущество соперника 3v2 в зоне высокого xG»
  • Игрок в зоне риска травмы → «Накоплено 42 спринта, скорость упала на 18%,
    ускорений за последние 15 мин: 0»
  • Тактическая ошибка → «Игрок #5 не закрыл зону между ЦЗ и фулбеком,
    что создало прямой коридор для прохода»
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Language = Literal["ru", "en"]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    """Объяснение одного предсказания или аналитического результата."""
    subject: str           # что объясняем
    verdict: str           # итоговый вывод (одна строка)
    factors: list[str]     # список факторов (почему такой вывод)
    confidence: float      # уверенность 0..1
    language: Language
    severity: Literal["info", "warning", "critical"] = "info"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "verdict": self.verdict,
            "factors": self.factors,
            "confidence": round(self.confidence, 3),
            "language": self.language,
            "severity": self.severity,
        }

    def to_text(self) -> str:
        """Форматированный текст объяснения."""
        lines = [f"[{self.subject}] {self.verdict}"]
        for i, f in enumerate(self.factors, 1):
            lines.append(f"  {i}. {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class Explainer:
    """
    Генерирует объяснения для всех типов аналитики FIE.

    Работает полностью локально, без API — rule-based объяснения
    на основе числовых данных.
    """

    def __init__(self, language: Language = "ru") -> None:
        self.language = language

    # ── Goal probability ───────────────────────────────────────────────────

    def explain_goal_probability(
        self,
        probability: float,
        *,
        compactness: float | None = None,
        distance_to_goal: float | None = None,
        players_between: int | None = None,
        shot_angle_deg: float | None = None,
        under_pressure: bool = False,
        xg_context: float | None = None,
    ) -> Explanation:
        """Объяснить вероятность гола."""
        factors: list[str] = []
        ru = self.language == "ru"

        pct = round(probability * 100, 1)

        # Компактность обороны
        if compactness is not None:
            if compactness < 0.35:
                factors.append(
                    "Оборона потеряла компактность — большие разрывы между линиями" if ru
                    else "Defense lost compactness — large gaps between lines"
                )
            elif compactness > 0.65:
                factors.append(
                    "Оборона компактна — минимум пространства для удара" if ru
                    else "Defense is compact — minimal space for a shot"
                )

        # Дистанция до ворот
        if distance_to_goal is not None:
            if distance_to_goal < 11:
                factors.append(
                    f"Удар с близкой дистанции ({distance_to_goal:.1f}м) — высокая опасность" if ru
                    else f"Close-range shot ({distance_to_goal:.1f}m) — high danger"
                )
            elif distance_to_goal > 25:
                factors.append(
                    f"Удар с дальней дистанции ({distance_to_goal:.1f}м) — низкая точность" if ru
                    else f"Long-range shot ({distance_to_goal:.1f}m) — low accuracy"
                )

        # Игроки между мячом и воротами
        if players_between is not None:
            if players_between == 0:
                factors.append(
                    "Нет защитников между мячом и воротами" if ru
                    else "No defenders between ball and goal"
                )
            elif players_between >= 3:
                factors.append(
                    f"Путь к воротам перекрыт {players_between} защитниками" if ru
                    else f"Path to goal blocked by {players_between} defenders"
                )

        # Угол удара
        if shot_angle_deg is not None:
            if shot_angle_deg < 15:
                factors.append(
                    f"Острый угол удара ({shot_angle_deg:.0f}°) — цель сильно сужена" if ru
                    else f"Acute shot angle ({shot_angle_deg:.0f}°) — target heavily reduced"
                )
            elif shot_angle_deg > 45:
                factors.append(
                    f"Хороший угол удара ({shot_angle_deg:.0f}°) — широкая цель" if ru
                    else f"Good shot angle ({shot_angle_deg:.0f}°) — wide target"
                )

        # Давление на бьющего
        if under_pressure:
            factors.append(
                "Игрок бьёт под давлением защитника — снижает точность" if ru
                else "Player shoots under defensive pressure — reduces accuracy"
            )

        # xG контекст
        if xg_context is not None:
            factors.append(
                f"xG модель оценивает момент в {xg_context:.2f}" if ru
                else f"xG model rates this chance at {xg_context:.2f}"
            )

        if not factors:
            factors.append(
                "Статистическая оценка на основе позиции и контекста игры" if ru
                else "Statistical assessment based on position and game context"
            )

        # Вердикт
        if probability >= 0.5:
            verdict = (f"ВЫСОКАЯ вероятность гола: {pct}%" if ru else f"HIGH goal probability: {pct}%")
            severity = "critical"
        elif probability >= 0.25:
            verdict = (f"СРЕДНЯЯ вероятность гола: {pct}%" if ru else f"MEDIUM goal probability: {pct}%")
            severity = "warning"
        else:
            verdict = (f"Низкая вероятность гола: {pct}%" if ru else f"Low goal probability: {pct}%")
            severity = "info"

        return Explanation(
            subject="Вероятность гола" if ru else "Goal Probability",
            verdict=verdict,
            factors=factors,
            confidence=min(0.95, 0.5 + len(factors) * 0.08),
            language=self.language,
            severity=severity,
        )

    # ── Fatigue / injury risk ──────────────────────────────────────────────

    def explain_fatigue(
        self,
        fatigue_score: float,
        *,
        player_id: int | None = None,
        sprint_count: int | None = None,
        speed_drop_pct: float | None = None,
        minutes_played: float | None = None,
        accel_count: int | None = None,
    ) -> Explanation:
        """Объяснить уровень усталости и риск травмы."""
        factors: list[str] = []
        ru = self.language == "ru"
        pid_str = f"#{player_id} " if player_id else ""

        if sprint_count is not None:
            limit = 35
            if sprint_count > limit:
                factors.append(
                    f"Накоплено {sprint_count} спринтов — выше нормы ({limit})" if ru
                    else f"Accumulated {sprint_count} sprints — above norm ({limit})"
                )
            else:
                factors.append(
                    f"Спринтов: {sprint_count} (в норме)" if ru
                    else f"Sprints: {sprint_count} (normal)"
                )

        if speed_drop_pct is not None and speed_drop_pct > 0:
            factors.append(
                f"Скорость упала на {speed_drop_pct:.1f}% относительно начала матча" if ru
                else f"Speed dropped by {speed_drop_pct:.1f}% vs match start"
            )

        if minutes_played is not None:
            factors.append(
                f"Проведено на поле: {minutes_played:.0f} мин" if ru
                else f"Minutes on field: {minutes_played:.0f}"
            )

        if accel_count is not None and accel_count > 30:
            factors.append(
                f"Высокое число резких ускорений: {accel_count} — нагрузка на мышцы" if ru
                else f"High explosive accelerations: {accel_count} — muscular load"
            )

        if not factors:
            factors.append(
                f"Оценка усталости на основе нагрузочных паттернов: {fatigue_score:.0f}/100" if ru
                else f"Fatigue assessed from load patterns: {fatigue_score:.0f}/100"
            )

        if fatigue_score >= 75:
            verdict = (f"Игрок {pid_str}в критической зоне усталости ({fatigue_score:.0f}/100)" if ru
                       else f"Player {pid_str}in critical fatigue zone ({fatigue_score:.0f}/100)")
            severity = "critical"
        elif fatigue_score >= 50:
            verdict = (f"Игрок {pid_str}устал ({fatigue_score:.0f}/100) — риск замены" if ru
                       else f"Player {pid_str}fatigued ({fatigue_score:.0f}/100) — substitution risk")
            severity = "warning"
        else:
            verdict = (f"Игрок {pid_str}в норме ({fatigue_score:.0f}/100)" if ru
                       else f"Player {pid_str}within normal range ({fatigue_score:.0f}/100)")
            severity = "info"

        return Explanation(
            subject=f"Усталость игрока {pid_str}" if ru else f"Player {pid_str}Fatigue",
            verdict=verdict,
            factors=factors,
            confidence=0.82,
            language=self.language,
            severity=severity,
        )

    # ── Tactical mistake ───────────────────────────────────────────────────

    def explain_tactical_mistake(
        self,
        mistake_type: str,
        *,
        player_id: int | None = None,
        severity: str = "medium",
        zone: str | None = None,
        minute: int | None = None,
        context: str | None = None,
    ) -> Explanation:
        """Объяснить тактическую ошибку."""
        ru = self.language == "ru"
        pid_str = f"#{player_id} " if player_id else ""
        min_str = f" на {minute}-й мин" if minute else ""

        mistake_explanations_ru = {
            "defensive_gap": (
                "Образовался разрыв в обороне — игрок не закрыл зону между линиями",
                ["Соперник получил свободное пространство для развития атаки",
                 "Нарушена компактность блока обороны",
                 "Требуется более тесное взаимодействие с ближайшими игроками"]
            ),
            "pressing_failure": (
                "Игрок не поддержал прессинг — соперник вышел из-под давления",
                ["Нарушена синхронность прессинга команды",
                 "Соперник получил лишнее время на передачу",
                 "Необходимо отработать командный прессинг на тренировке"]
            ),
            "late_recovery": (
                "Запоздалое возвращение в оборону после атаки",
                ["Создан численный перевес соперника в контратаке",
                 "Игрок оказался вне позиции при быстром переходе",
                 "Требуется улучшить реакцию на потерю мяча"]
            ),
            "open_corridor": (
                "Открыт коридор для прохода соперника",
                ["Неправильная начальная позиция относительно мяча",
                 "Соперник воспользовался пространством за спиной",
                 "Необходимо контролировать глубину позиции"]
            ),
            "line_break": (
                "Нарушена линия обороны — выбился из единого блока",
                ["Ошибка в прочтении игровой ситуации",
                 "Разрыв между линиями обороны и полузащиты увеличился",
                 "Рекомендуется отработка синхронного движения линии"]
            ),
        }

        mistake_explanations_en = {
            "defensive_gap": (
                "Defensive gap formed — player failed to cover the zone between lines",
                ["Opponent gained free space for attacking development",
                 "Block compactness compromised",
                 "Closer coordination with nearby players required"]
            ),
            "pressing_failure": (
                "Player did not support pressing — opponent escaped pressure",
                ["Team pressing synchrony broken",
                 "Opponent had extra time on the ball",
                 "Team pressing needs to be drilled in training"]
            ),
            "late_recovery": (
                "Late defensive recovery after attack",
                ["Numerical advantage created for opponent on counter",
                 "Player caught out of position on quick transition",
                 "Reaction to ball loss needs improvement"]
            ),
            "open_corridor": (
                "Corridor opened for opponent to advance",
                ["Incorrect starting position relative to ball",
                 "Opponent exploited space behind the player",
                 "Positional depth awareness needs work"]
            ),
            "line_break": (
                "Defensive line broken — player stepped out of block",
                ["Misread of game situation",
                 "Gap between defensive and midfield lines widened",
                 "Synchronized line movement needs drilling"]
            ),
        }

        exps = mistake_explanations_ru if ru else mistake_explanations_en
        default_verdict = (f"Тактическая ошибка: {mistake_type}" if ru
                           else f"Tactical mistake: {mistake_type}")
        default_factors = [context] if context else [
            ("Ситуационный анализ недоступен" if ru else "Situational analysis unavailable")
        ]

        verdict_tmpl, factors = exps.get(mistake_type, (default_verdict, default_factors))
        verdict = f"Игрок {pid_str}{min_str}: {verdict_tmpl}" if (pid_str or min_str) else verdict_tmpl

        if zone:
            factors = list(factors) + [
                (f"Зона нарушения: {zone}" if ru else f"Zone of violation: {zone}")
            ]

        sev_map = {"high": "critical", "medium": "warning", "low": "info"}

        return Explanation(
            subject=("Тактическая ошибка" if ru else "Tactical Mistake"),
            verdict=verdict,
            factors=list(factors),
            confidence=0.78,
            language=self.language,
            severity=sev_map.get(severity, "warning"),
        )

    # ── Match prediction ───────────────────────────────────────────────────

    def explain_match_prediction(
        self,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        *,
        home_team: str = "Home",
        away_team: str = "Away",
        key_factors: list[str] | None = None,
        home_xg: float | None = None,
        away_xg: float | None = None,
        home_fatigue_avg: float | None = None,
        away_fatigue_avg: float | None = None,
    ) -> Explanation:
        """Объяснить предсказание исхода матча."""
        ru = self.language == "ru"
        factors: list[str] = []

        # Ключевые факторы из модели
        if key_factors:
            factors.extend(key_factors)

        # xG
        if home_xg is not None and away_xg is not None:
            factors.append(
                f"xG: {home_team} {home_xg:.2f} vs {away_team} {away_xg:.2f}" if ru
                else f"xG: {home_team} {home_xg:.2f} vs {away_team} {away_xg:.2f}"
            )

        # Усталость
        if home_fatigue_avg is not None and away_fatigue_avg is not None:
            if abs(home_fatigue_avg - away_fatigue_avg) > 10:
                more_tired = home_team if home_fatigue_avg > away_fatigue_avg else away_team
                factors.append(
                    f"Команда {more_tired} более измотана физически" if ru
                    else f"Team {more_tired} is more physically fatigued"
                )

        if not factors:
            factors.append(
                "Модель оценивает исход на основе исторических данных и текущего контекста" if ru
                else "Model evaluates outcome based on historical data and current context"
            )

        # Вердикт
        probs = {home_team: home_win_prob, "draw": draw_prob, away_team: away_win_prob}
        winner = max(probs, key=lambda k: probs[k])
        max_prob = probs[winner]

        if winner == "draw":
            verdict = (f"Наиболее вероятна ничья ({draw_prob*100:.0f}%)" if ru
                       else f"Draw most likely ({draw_prob*100:.0f}%)")
        else:
            verdict = (f"Победа {winner} наиболее вероятна ({max_prob*100:.0f}%)" if ru
                       else f"{winner} win most likely ({max_prob*100:.0f}%)")

        severity = "info" if max_prob < 0.55 else ("warning" if max_prob < 0.75 else "critical")

        return Explanation(
            subject=f"{home_team} vs {away_team}",
            verdict=verdict,
            factors=factors,
            confidence=min(0.9, max_prob + 0.1),
            language=self.language,
            severity=severity,
        )

    # ── Team DNA ───────────────────────────────────────────────────────────

    def explain_team_dna(
        self,
        dna: dict,
        *,
        team_name: str = "Team",
        compare_with: dict | None = None,
        compare_name: str = "Opponent",
    ) -> Explanation:
        """Объяснить тактический профиль команды (Team DNA)."""
        ru = self.language == "ru"
        factors: list[str] = []

        pi = dna.get("pressing_intensity", 0.5)
        pl = dna.get("pressing_line", 0.5)
        tempo = dna.get("tempo", 0.5)
        terr = dna.get("territory", 0.5)
        width = dna.get("attack_width", 0.5)
        agg = dna.get("aggression", 0.5)

        # Прессинг
        if pi > 0.65:
            factors.append("Высокоинтенсивный прессинг — команда активно давит на соперника" if ru
                           else "High-intensity pressing — team aggressively pressures opponents")
        elif pi < 0.35:
            factors.append("Пассивная оборона — команда отходит в низкий блок" if ru
                           else "Passive defense — team sits in a low block")

        # Линия прессинга
        if pl > 0.6:
            factors.append("Высокая линия обороны — прессинг начинается на половине соперника" if ru
                           else "High defensive line — pressing starts in opponent's half")
        elif pl < 0.4:
            factors.append("Низкий оборонительный блок — защита у своей штрафной" if ru
                           else "Low defensive block — defending close to own box")

        # Темп
        if tempo > 0.65:
            factors.append("Высокий темп игры — быстрые переходы и вертикальные атаки" if ru
                           else "High tempo — fast transitions and vertical attacks")

        # Территориальный контроль
        if terr > 0.6:
            factors.append("Территориальное доминирование — команда контролирует половину соперника" if ru
                           else "Territorial dominance — team controls opponent's half")
        elif terr < 0.4:
            factors.append("Оборонительная территориальность — команда уступает пространство" if ru
                           else "Defensive territory — team cedes space")

        # Ширина атаки
        if width > 0.65:
            factors.append("Широкая атака — активное использование флангов" if ru
                           else "Wide attack — active use of flanks")

        # Агрессивность
        if agg > 0.7:
            factors.append("Высокая агрессивность — много единоборств и жёсткая игра" if ru
                           else "High aggression — many duels and physical play")

        # Сравнение
        if compare_with:
            sim = compare_with.get("similarity", None)
            if sim is not None:
                if sim > 0.8:
                    factors.append(
                        f"Стиль очень похож на {compare_name} (схожесть {sim:.0%})" if ru
                        else f"Style very similar to {compare_name} (similarity {sim:.0%})"
                    )
                elif sim < 0.4:
                    factors.append(
                        f"Стиль кардинально отличается от {compare_name} (схожесть {sim:.0%})" if ru
                        else f"Style very different from {compare_name} (similarity {sim:.0%})"
                    )

        if not factors:
            factors.append("Тактический профиль в среднем диапазоне по всем показателям" if ru
                           else "Tactical profile in average range across all metrics")

        verdict = (f"Тактический ДНК команды {team_name}: {_dna_style_label(pi, pl, tempo, ru)}" if ru
                   else f"Team DNA of {team_name}: {_dna_style_label(pi, pl, tempo, ru)}")

        return Explanation(
            subject=f"Team DNA — {team_name}",
            verdict=verdict,
            factors=factors,
            confidence=0.85,
            language=self.language,
            severity="info",
        )

    # ── Scouting ───────────────────────────────────────────────────────────

    def explain_scouting(
        self,
        talent_score: float,
        *,
        player_id: int | None = None,
        position: str | None = None,
        strengths: list[str] | None = None,
        weaknesses: list[str] | None = None,
        readiness_pct: float | None = None,
    ) -> Explanation:
        """Объяснить скаутинговую оценку игрока."""
        ru = self.language == "ru"
        pid_str = f"#{player_id} " if player_id else ""
        pos_str = f"({position})" if position else ""
        factors: list[str] = []

        if strengths:
            for s in strengths[:3]:
                factors.append(f"✓ {s}" if not ru else f"✓ {s}")

        if weaknesses:
            for w in weaknesses[:2]:
                factors.append(f"✗ {w}" if not ru else f"✗ {w}")

        if readiness_pct is not None:
            factors.append(
                f"Готовность к переходу: {readiness_pct:.0f}%" if ru
                else f"Transfer readiness: {readiness_pct:.0f}%"
            )

        if not factors:
            factors.append(
                f"Скаутинговая оценка: {talent_score:.0f}/100" if ru
                else f"Scouting score: {talent_score:.0f}/100"
            )

        if talent_score >= 80:
            verdict = (f"Игрок {pid_str}{pos_str} — ВЫСОКИЙ потенциал ({talent_score:.0f}/100)" if ru
                       else f"Player {pid_str}{pos_str} — HIGH potential ({talent_score:.0f}/100)")
            severity = "critical"
        elif talent_score >= 60:
            verdict = (f"Игрок {pid_str}{pos_str} — перспективный ({talent_score:.0f}/100)" if ru
                       else f"Player {pid_str}{pos_str} — promising ({talent_score:.0f}/100)")
            severity = "warning"
        else:
            verdict = (f"Игрок {pid_str}{pos_str} — требует развития ({talent_score:.0f}/100)" if ru
                       else f"Player {pid_str}{pos_str} — needs development ({talent_score:.0f}/100)")
            severity = "info"

        return Explanation(
            subject=("Скаутинг игрока" if ru else "Player Scouting"),
            verdict=verdict,
            factors=factors,
            confidence=0.75,
            language=self.language,
            severity=severity,
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dna_style_label(pi: float, pl: float, tempo: float, ru: bool) -> str:
    if pi > 0.65 and pl > 0.55:
        return "Высокий прессинг / Gegenpressing" if ru else "High Press / Gegenpressing"
    if pi < 0.4 and pl < 0.45:
        return "Низкий блок / Контратаки" if ru else "Low Block / Counter-Attack"
    if tempo > 0.65:
        return "Вертикальный быстрый футбол" if ru else "Vertical Fast Football"
    return "Позиционный контроль" if ru else "Positional Control"
