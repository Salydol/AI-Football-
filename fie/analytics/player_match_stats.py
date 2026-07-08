"""
Player Match Statistics — секции 32-44 продуктовой спецификации.

Полная модель PlayerMatchStat со всеми вкладками:
  General / Attacking / Defending / Passing / Duels / Goalkeeping

Дополнительно:
  - AI Notes (section 42) — авто-заметки по игроку
  - AI Rankings (section 41) — лучший/худший по категориям
  - Performance Overlay data (section 40)
  - Player Match Card (section 39)
  - Player Comparison (section 43)
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass
class PlayerMatchStat:
    """Full per-player match statistics (section 44 — PlayerMatchStat entity)."""

    # --- Identity ---
    player_id: str
    player_name: str
    team_id: str
    match_id: str
    shirt_number: int
    position: str          # GK CB LB RB CDM CM CAM LW RW ST
    minutes_played: int
    is_starter: bool

    # --- General ---
    goals: int = 0
    assists: int = 0
    rating: float = 0.0
    yellow_cards: int = 0
    red_cards: int = 0
    is_captain: bool = False
    substituted_in: Optional[int] = None   # minute entered
    substituted_out: Optional[int] = None  # minute exited

    # --- Attacking ---
    shots_on_target: int = 0
    shots_off_target: int = 0
    shots_blocked: int = 0
    xg: float = 0.0
    xa: float = 0.0
    xgot: float = 0.0
    key_passes: int = 0
    big_chances_missed: int = 0
    big_chances_created: int = 0
    hit_woodwork: int = 0
    touches_in_box: int = 0

    # --- Defending ---
    tackles_won: int = 0
    total_tackles: int = 0
    interceptions: int = 0
    clearances: int = 0
    blocked_shots: int = 0
    defensive_actions: int = 0
    recoveries: int = 0
    dribbled_past: int = 0
    errors_leading_to_shot: int = 0
    errors_leading_to_goal: int = 0
    clearances_off_line: int = 0

    # --- Passing ---
    accurate_passes: int = 0
    total_passes: int = 0
    accurate_crosses: int = 0
    total_crosses: int = 0
    accurate_long_balls: int = 0
    total_long_balls: int = 0
    progressive_passes: int = 0
    passes_into_final_third: int = 0
    passes_into_box: int = 0
    build_up_involvement: float = 0.0

    # --- Duels ---
    duels_won: int = 0
    total_duels: int = 0
    ground_duels_won: int = 0
    total_ground_duels: int = 0
    aerial_duels_won: int = 0
    total_aerial_duels: int = 0
    possession_lost: int = 0
    fouls: int = 0
    was_fouled: int = 0
    offsides: int = 0

    # --- Touches / Dribbling ---
    touches: int = 0
    touches_in_final_third: int = 0
    successful_dribbles: int = 0
    total_dribbles: int = 0
    carries: int = 0
    progressive_carries: int = 0
    ball_retention: float = 0.0
    press_resistance: float = 0.0

    # --- Goalkeeping (GK only) ---
    goalkeeper_saves: int = 0
    goals_prevented: float = 0.0
    punches: int = 0
    high_claims: int = 0
    saves_from_inside_box: int = 0
    saves_from_outside_box: int = 0
    big_saves: int = 0
    xgot_faced: float = 0.0
    goals_conceded: int = 0
    goal_kicks: int = 0
    gk_pass_accuracy: float = 0.0
    gk_long_balls: int = 0
    sweeper_actions: int = 0

    # --- AI output ---
    notes: str = ""
    ai_summary: str = ""

    # --- AI Indices ---
    shot_impact_index: float = 0.0
    finishing_quality_index: float = 0.0
    chance_creation_index: float = 0.0
    attacking_threat_score: float = 0.0
    defensive_reliability_index: float = 0.0
    zone_protection_score: float = 0.0
    build_up_quality_index: float = 0.0
    physical_dominance_index: float = 0.0
    goalkeeper_impact_score: float = 0.0
    save_difficulty_index: float = 0.0

    # --- Pass completion % (computed) ---
    @property
    def pass_completion_pct(self) -> float:
        return round(self.accurate_passes / self.total_passes * 100, 1) if self.total_passes else 0.0

    @property
    def duel_win_pct(self) -> float:
        return round(self.duels_won / self.total_duels * 100, 1) if self.total_duels else 0.0

    @property
    def total_shots(self) -> int:
        return self.shots_on_target + self.shots_off_target + self.shots_blocked

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pass_completion_pct"] = self.pass_completion_pct
        d["duel_win_pct"] = self.duel_win_pct
        d["total_shots"] = self.total_shots
        return d


# ---------------------------------------------------------------------------
# Stats generator (rule-based, realistic distributions per position)
# ---------------------------------------------------------------------------

def _rng_seed(player_id: str, match_id: str) -> random.Random:
    """Deterministic RNG from player+match IDs so results are stable."""
    seed = hash(player_id + "|" + match_id) & 0xFFFFFFFF
    return random.Random(seed)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


_POSITION_PROFILES: dict[str, dict] = {
    "GK": dict(passes=(25, 40), tackles=(0, 1), duels=(2, 5), shots=(0, 0)),
    "CB": dict(passes=(40, 65), tackles=(2, 6), duels=(5, 12), shots=(0, 1)),
    "LB": dict(passes=(40, 70), tackles=(2, 5), duels=(5, 10), shots=(0, 2)),
    "RB": dict(passes=(40, 70), tackles=(2, 5), duels=(5, 10), shots=(0, 2)),
    "CDM": dict(passes=(50, 80), tackles=(3, 7), duels=(6, 14), shots=(0, 2)),
    "CM": dict(passes=(45, 75), tackles=(2, 5), duels=(5, 12), shots=(1, 3)),
    "CAM": dict(passes=(40, 65), tackles=(1, 3), duels=(4, 10), shots=(1, 4)),
    "LW": dict(passes=(30, 55), tackles=(1, 3), duels=(4, 10), shots=(1, 5)),
    "RW": dict(passes=(30, 55), tackles=(1, 3), duels=(4, 10), shots=(1, 5)),
    "ST": dict(passes=(20, 45), tackles=(0, 2), duels=(3, 8), shots=(2, 6)),
}


def generate_player_stats(
    player_id: str,
    player_name: str,
    team_id: str,
    match_id: str,
    shirt_number: int,
    position: str,
    minutes_played: int = 90,
    is_starter: bool = True,
    goals: int = 0,
    assists: int = 0,
    yellow_cards: int = 0,
    red_cards: int = 0,
    is_captain: bool = False,
    substituted_in: Optional[int] = None,
    substituted_out: Optional[int] = None,
) -> PlayerMatchStat:
    """
    Generate realistic rule-based stats for one player in a match.
    In production, replace with data from the tracking / event pipeline.
    """
    rng = _rng_seed(player_id, match_id)
    prof = _POSITION_PROFILES.get(position, _POSITION_PROFILES["CM"])
    scale = minutes_played / 90.0

    # --- Passing ---
    total_passes = int(rng.randint(*prof["passes"]) * scale)
    pass_acc_pct = rng.uniform(0.72, 0.93)
    accurate_passes = int(total_passes * pass_acc_pct)
    total_crosses = int(rng.randint(0, 5 if position in ("LW","RW","LB","RB") else 1) * scale)
    accurate_crosses = int(total_crosses * rng.uniform(0.2, 0.6))
    total_long = int(rng.randint(0, 8 if position in ("GK","CB","CDM") else 3) * scale)
    accurate_long = int(total_long * rng.uniform(0.4, 0.75))
    prog_passes = int(rng.randint(1, 12) * scale)
    final3_passes = int(prog_passes * rng.uniform(0.3, 0.7))
    box_passes = int(rng.randint(0, 4 if position in ("CAM","LW","RW","CM") else 1) * scale)
    key_passes = int(rng.randint(0, 3) * scale)
    build_up = round(rng.uniform(0.3, 0.9), 2)

    # --- Shooting ---
    total_sh = int(rng.randint(*prof["shots"]) * scale)
    sot = min(total_sh, int(total_sh * rng.uniform(0.3, 0.7)))
    soff = max(0, total_sh - sot - rng.randint(0, max(1, total_sh // 3)))
    sblk = total_sh - sot - soff
    xg_per_shot = rng.uniform(0.05, 0.25)
    xg = round(total_sh * xg_per_shot, 2)
    xgot = round(sot * rng.uniform(0.08, 0.35), 2)
    bchm = int(rng.random() < 0.15) if total_sh > 2 else 0
    bchc = int(rng.random() < 0.12)
    woodwork = int(rng.random() < 0.05)
    touches_box = int(rng.randint(0, 8) * scale) if position in ("ST","CAM","LW","RW") else int(rng.randint(0, 3) * scale)
    xa = round(key_passes * rng.uniform(0.05, 0.2), 2)
    xgot_v = round(xg * rng.uniform(0.8, 1.4), 2) if position == "GK" else 0.0

    # --- Defending ---
    total_tack = int(rng.randint(*prof["tackles"]) * scale)
    tack_won = int(total_tack * rng.uniform(0.5, 0.85))
    interc = int(rng.randint(0, 4 if position in ("CB","CDM","CM","LB","RB") else 1) * scale)
    clearances = int(rng.randint(0, 6 if position in ("CB","GK") else 2) * scale)
    blk_shots = int(rng.randint(0, 3) * scale)
    def_actions = tack_won + interc + clearances + blk_shots
    recoveries = int(rng.randint(2, 10) * scale)
    dribbled_past = int(rng.randint(0, 3) * scale)
    err_shot = int(rng.random() < 0.08)
    err_goal = int(rng.random() < 0.03)
    coff_line = int(rng.random() < 0.05)

    # --- Duels ---
    total_dur = int(rng.randint(*prof["duels"]) * scale)
    total_ground = int(total_dur * rng.uniform(0.6, 0.85))
    total_aerial = total_dur - total_ground
    ground_won = int(total_ground * rng.uniform(0.4, 0.75))
    aerial_won = int(total_aerial * rng.uniform(0.35, 0.70))
    duels_won_total = ground_won + aerial_won
    poss_lost = int(rng.randint(3, 18) * scale)
    fouls = int(rng.randint(0, 3) * scale)
    was_fouled = int(rng.randint(0, 4) * scale)
    offsides = int(rng.randint(0, 2) * scale) if position in ("ST","LW","RW","CAM") else 0

    # --- Touches / Dribbling ---
    touches = int(rng.randint(30, 90) * scale)
    t_final3 = int(touches * rng.uniform(0.1, 0.4))
    total_drib = int(rng.randint(0, 6) * scale)
    succ_drib = int(total_drib * rng.uniform(0.4, 0.75))
    carries = int(rng.randint(5, 25) * scale)
    prog_carries = int(carries * rng.uniform(0.2, 0.5))
    ball_ret = round(rng.uniform(0.55, 0.88), 2)
    press_res = round(rng.uniform(0.40, 0.85), 2)

    # --- Goalkeeping ---
    gk_saves = gk_prev = gk_punch = gk_high = 0
    gk_in = gk_out = gk_big = gk_xgot = gk_conc = gk_kicks = 0
    gk_pass_acc = gk_long_b = gk_swp = 0
    if position == "GK":
        gk_conc = int(rng.randint(0, 3))
        shots_faced = gk_conc + rng.randint(1, 6)
        gk_saves = shots_faced - gk_conc
        gk_in = int(gk_saves * rng.uniform(0.5, 0.8))
        gk_out = gk_saves - gk_in
        gk_big = int(rng.randint(0, 2))
        gk_xgot = round(shots_faced * rng.uniform(0.08, 0.22), 2)
        gk_prev = round(gk_xgot - gk_conc * rng.uniform(0.05, 0.15), 2)
        gk_punch = int(rng.randint(0, 2))
        gk_high = int(rng.randint(0, 3))
        gk_kicks = int(rng.randint(10, 25))
        gk_pass_acc = round(rng.uniform(0.55, 0.85), 2)
        gk_long_b = int(rng.randint(5, 18))
        gk_swp = int(rng.randint(0, 4))
        total_passes = total_passes  # keep calculated passes

    # --- Compute AI Indices ---
    shot_impact = round(_clamp((sot + xg * 2) / 5.0, 0, 1), 2)
    finish_q = round(_clamp(xgot / max(xg, 0.01), 0, 1.5), 2) if xg > 0 else 0.0
    chance_cr = round(_clamp((key_passes * 0.3 + bchc * 0.5 + xa * 2) / 3.0, 0, 1), 2)
    atk_threat = round(_clamp((xg * 3 + bchm * 0.3 + sot * 0.2) / 2.0, 0, 1), 2)
    def_rel = round(_clamp((tack_won * 0.4 + interc * 0.3 + clearances * 0.2 - err_shot * 0.5) / 4.0, 0, 1), 2)
    zone_prot = round(_clamp(1.0 - dribbled_past * 0.15 - err_shot * 0.3, 0, 1), 2)
    buq = round(_clamp((prog_passes * 0.3 + accurate_passes * 0.01 + build_up) / 1.5, 0, 1), 2)
    phys_dom = round(_clamp((ground_won * 0.4 + aerial_won * 0.5) / max(total_dur, 1), 0, 1), 2)
    gk_impact = round(_clamp((gk_saves * 0.3 + gk_prev) / max(gk_xgot + 0.01, 0.1), 0, 2), 2) if position == "GK" else 0.0
    save_diff = round(_clamp(gk_xgot / max(gk_saves + 0.5, 0.5), 0, 1), 2) if position == "GK" else 0.0

    # --- Rating ---
    rating = _compute_rating(
        position=position, goals=goals, assists=assists,
        xg=xg, xa=xa, accurate_passes=accurate_passes, total_passes=total_passes,
        def_actions=def_actions, duels_won=duels_won_total, total_duels=total_dur,
        err_shot=err_shot, err_goal=err_goal, gk_saves=gk_saves, gk_conc=gk_conc,
        minutes=minutes_played,
    )

    # --- AI Notes ---
    notes = _generate_notes(
        position=position, goals=goals, assists=assists, rating=rating,
        bchm=bchm, bchc=bchc, woodwork=woodwork,
        err_shot=err_shot, err_goal=err_goal, coff_line=coff_line,
        gk_big=gk_big, gk_saves=gk_saves, gk_conc=gk_conc,
        dribbled_past=dribbled_past, poss_lost=poss_lost,
        interc=interc, tack_won=tack_won,
        minutes_played=minutes_played,
    )

    # --- AI Summary ---
    ai_summary = _generate_ai_summary(
        player_name=player_name, position=position, rating=rating,
        goals=goals, assists=assists, xg=xg, xa=xa,
        accurate_passes=accurate_passes, total_passes=total_passes,
        def_actions=def_actions, notes=notes,
        minutes_played=minutes_played,
    )

    return PlayerMatchStat(
        player_id=player_id, player_name=player_name,
        team_id=team_id, match_id=match_id,
        shirt_number=shirt_number, position=position,
        minutes_played=minutes_played, is_starter=is_starter,
        goals=goals, assists=assists, rating=rating,
        yellow_cards=yellow_cards, red_cards=red_cards,
        is_captain=is_captain,
        substituted_in=substituted_in, substituted_out=substituted_out,
        # attacking
        shots_on_target=sot, shots_off_target=soff, shots_blocked=sblk,
        xg=xg, xa=xa, xgot=xgot,
        key_passes=key_passes, big_chances_missed=bchm,
        big_chances_created=bchc, hit_woodwork=woodwork,
        touches_in_box=touches_box,
        # defending
        tackles_won=tack_won, total_tackles=total_tack,
        interceptions=interc, clearances=clearances,
        blocked_shots=blk_shots, defensive_actions=def_actions,
        recoveries=recoveries, dribbled_past=dribbled_past,
        errors_leading_to_shot=err_shot, errors_leading_to_goal=err_goal,
        clearances_off_line=coff_line,
        # passing
        accurate_passes=accurate_passes, total_passes=total_passes,
        accurate_crosses=accurate_crosses, total_crosses=total_crosses,
        accurate_long_balls=accurate_long, total_long_balls=total_long,
        progressive_passes=prog_passes,
        passes_into_final_third=final3_passes, passes_into_box=box_passes,
        build_up_involvement=build_up,
        # duels
        duels_won=duels_won_total, total_duels=total_dur,
        ground_duels_won=ground_won, total_ground_duels=total_ground,
        aerial_duels_won=aerial_won, total_aerial_duels=total_aerial,
        possession_lost=poss_lost, fouls=fouls,
        was_fouled=was_fouled, offsides=offsides,
        # touches/dribbling
        touches=touches, touches_in_final_third=t_final3,
        successful_dribbles=succ_drib, total_dribbles=total_drib,
        carries=carries, progressive_carries=prog_carries,
        ball_retention=ball_ret, press_resistance=press_res,
        # goalkeeping
        goalkeeper_saves=gk_saves, goals_prevented=gk_prev,
        punches=gk_punch, high_claims=gk_high,
        saves_from_inside_box=gk_in, saves_from_outside_box=gk_out,
        big_saves=gk_big, xgot_faced=gk_xgot,
        goals_conceded=gk_conc, goal_kicks=gk_kicks,
        gk_pass_accuracy=gk_pass_acc, gk_long_balls=gk_long_b,
        sweeper_actions=gk_swp,
        # notes / summary
        notes=notes, ai_summary=ai_summary,
        # indices
        shot_impact_index=shot_impact,
        finishing_quality_index=finish_q,
        chance_creation_index=chance_cr,
        attacking_threat_score=atk_threat,
        defensive_reliability_index=def_rel,
        zone_protection_score=zone_prot,
        build_up_quality_index=buq,
        physical_dominance_index=phys_dom,
        goalkeeper_impact_score=gk_impact,
        save_difficulty_index=save_diff,
    )


def _compute_rating(
    position: str, goals: int, assists: int, xg: float, xa: float,
    accurate_passes: int, total_passes: int,
    def_actions: int, duels_won: int, total_duels: int,
    err_shot: int, err_goal: int, gk_saves: int, gk_conc: int,
    minutes: int,
) -> float:
    base = 6.0
    scale = minutes / 90.0

    # Goals / assists boost
    base += goals * 1.2 + assists * 0.8

    # Pass accuracy
    if total_passes > 10:
        pct = accurate_passes / total_passes
        base += (pct - 0.78) * 3.0

    # Duels
    if total_duels > 0:
        base += (duels_won / total_duels - 0.5) * 1.2

    # xG contribution
    base += xg * 0.8 + xa * 0.5

    # Defensive actions
    base += min(def_actions, 8) * 0.08

    # Errors
    base -= err_shot * 0.4 + err_goal * 0.8

    # GK bonus
    if position == "GK":
        base += gk_saves * 0.3 - gk_conc * 0.4

    return round(_clamp(base, 4.0, 10.0), 1)


def _generate_notes(
    position: str, goals: int, assists: int, rating: float,
    bchm: int, bchc: int, woodwork: int,
    err_shot: int, err_goal: int, coff_line: int,
    gk_big: int, gk_saves: int, gk_conc: int,
    dribbled_past: int, poss_lost: int,
    interc: int, tack_won: int,
    minutes_played: int,
) -> str:
    parts = []
    if goals:
        parts.append(f"Goals: {goals}")
    if assists:
        parts.append(f"Assists: {assists}")
    if bchm:
        parts.append(f"Big chances missed: {bchm}")
    if bchc:
        parts.append(f"Big chances created: {bchc}")
    if woodwork:
        parts.append("Hit woodwork: 1")
    if err_goal:
        parts.append("Error leading to goal")
    elif err_shot:
        parts.append("Error leading to shot")
    if coff_line:
        parts.append("Clearance off line: 1")
    if gk_big:
        parts.append(f"Big saves: {gk_big}")
    if position == "GK" and gk_conc == 0 and gk_saves >= 3:
        parts.append("Clean sheet — key saves")
    if dribbled_past >= 3:
        parts.append("Dribbled past too often")
    if poss_lost >= 15:
        parts.append("Lost possession under pressure")
    if rating < 6.2:
        parts.append("Weak overall contribution")
    if tack_won >= 4:
        parts.append("High pressing impact")
    if interc >= 3:
        parts.append("Strong interception game")
    if minutes_played < 60:
        parts.append("Substituted early")
    if not parts:
        if rating >= 8.0:
            parts.append("Outstanding performance")
        elif rating >= 7.0:
            parts.append("Solid contribution")
        else:
            parts.append("Average display")
    return " · ".join(parts)


def _generate_ai_summary(
    player_name: str, position: str, rating: float,
    goals: int, assists: int, xg: float, xa: float,
    accurate_passes: int, total_passes: int,
    def_actions: int, notes: str,
    minutes_played: int,
) -> str:
    pct = f"{accurate_passes}/{total_passes}" if total_passes else "—"
    lines = [
        f"{player_name} ({position}) — рейтинг {rating}/10 за {minutes_played} мин.",
    ]
    if goals or assists:
        lines.append(f"Голевые действия: {goals} гол(а), {assists} ассист(а).")
    if xg > 0:
        lines.append(f"Создал xG={xg:.2f} (ожидаемые голы), xA={xa:.2f}.")
    lines.append(f"Точность паса: {pct}.")
    lines.append(f"Защитные действия: {def_actions}.")
    if notes:
        lines.append(f"Заметки: {notes}.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Performance Overlay (section 40)
# ---------------------------------------------------------------------------

OVERLAY_TYPES = ("rating", "shooting", "passing", "dribbling", "defending")


def build_overlay(stat: PlayerMatchStat, overlay_type: str, x: float, y: float) -> dict:
    """Return PlayerMetricOverlay dict for one player."""
    if overlay_type == "rating":
        primary = stat.rating
        secondary = stat.goals
        tertiary = stat.assists
        color = "green" if primary >= 7.5 else ("yellow" if primary >= 6.5 else "red")
        label = str(primary)
        tooltip = f"Рейтинг {primary}/10 — голы: {secondary}, ассисты: {tertiary}"

    elif overlay_type == "shooting":
        primary = stat.xg
        secondary = stat.xgot
        tertiary = stat.total_shots
        color = "red" if primary >= 0.3 else ("orange" if primary >= 0.1 else "gray")
        label = f"xG {primary:.2f}"
        tooltip = f"xG: {primary:.2f} · xGOT: {secondary:.2f} · удары: {tertiary}"
        if stat.big_chances_missed:
            tooltip += f" · Big chances missed: {stat.big_chances_missed}"

    elif overlay_type == "passing":
        primary = stat.xa
        secondary = stat.accurate_passes
        tertiary = stat.total_passes
        color = "blue" if stat.pass_completion_pct >= 85 else ("lightblue" if stat.pass_completion_pct >= 75 else "gray")
        label = f"xA {primary:.2f}"
        tooltip = (f"xA: {primary:.2f} · пасы: {secondary}/{tertiary} "
                   f"({stat.pass_completion_pct}%) · key passes: {stat.key_passes}")

    elif overlay_type == "dribbling":
        primary = stat.successful_dribbles
        secondary = stat.touches
        tertiary = stat.possession_lost
        color = "purple" if primary >= 3 else ("violet" if primary >= 1 else "gray")
        label = f"DR {primary}"
        tooltip = (f"Дриблинг: {primary}/{stat.total_dribbles} · "
                   f"касания: {secondary} · потери: {tertiary}")

    else:  # defending
        primary = stat.defensive_actions
        secondary = stat.tackles_won
        tertiary = stat.interceptions
        color = "darkgreen" if primary >= 8 else ("green" if primary >= 4 else "gray")
        label = f"DC {primary}"
        tooltip = (f"Защитные действия: {primary} · отборы: {secondary} · "
                   f"перехваты: {tertiary} · ошибки: {stat.errors_leading_to_shot}")

    return {
        "player_id": stat.player_id,
        "player_name": stat.player_name,
        "shirt_number": stat.shirt_number,
        "position": stat.position,
        "team_id": stat.team_id,
        "x_coordinate": round(x, 2),
        "y_coordinate": round(y, 2),
        "overlay_type": overlay_type,
        "primary_metric": primary,
        "secondary_metric": secondary,
        "tertiary_metric": tertiary,
        "color": color,
        "label": label,
        "tooltip": tooltip,
        "ai_explanation": stat.ai_summary,
    }


# ---------------------------------------------------------------------------
# AI Rankings (section 41)
# ---------------------------------------------------------------------------

def rank_players(stats: list[PlayerMatchStat]) -> dict:
    """Return best/worst per category from a list of PlayerMatchStat."""
    if not stats:
        return {}

    def _best(key_fn, label: str) -> dict | None:
        s = sorted(stats, key=key_fn, reverse=True)
        return {"player_id": s[0].player_id, "player_name": s[0].player_name,
                "value": key_fn(s[0]), "category": label} if s else None

    def _worst(key_fn, label: str) -> dict | None:
        s = sorted(stats, key=key_fn)
        return {"player_id": s[0].player_id, "player_name": s[0].player_name,
                "value": key_fn(s[0]), "category": label} if s else None

    outfield = [s for s in stats if s.position != "GK"]
    gks = [s for s in stats if s.position == "GK"]

    result = {
        "best_player": _best(lambda s: s.rating, "best_player"),
        "best_attacker": _best(lambda s: s.xg + s.xa + s.goals * 0.5, "best_attacker"),
        "best_passer": _best(lambda s: s.build_up_quality_index + s.pass_completion_pct / 100, "best_passer"),
        "best_dribbler": _best(lambda s: s.successful_dribbles, "best_dribbler"),
        "best_defender": _best(lambda s: s.defensive_reliability_index, "best_defender"),
        "best_off_ball": _best(lambda s: s.recoveries + s.interceptions, "best_off_ball"),
        "best_physical": _best(lambda s: s.physical_dominance_index, "best_physical"),
        "best_pressing": _best(lambda s: s.tackles_won + s.interceptions * 0.8, "best_pressing"),
        "worst_possession_loss": _worst(lambda s: -s.possession_lost, "worst_possession_loss"),
        "worst_position_discipline": _worst(lambda s: -s.dribbled_past, "worst_position_discipline"),
        "most_overloaded": _best(lambda s: s.touches + s.total_duels, "most_overloaded"),
        "substitution_risk": _best(lambda s: (90 - s.minutes_played) * 0.1 + (10 - s.rating), "substitution_risk"),
        "hidden_impact": _best(lambda s: s.recoveries * 0.4 + s.interceptions * 0.5 + s.progressive_carries * 0.2, "hidden_impact"),
    }
    if gks:
        result["best_goalkeeper"] = _best(lambda s: s.goalkeeper_impact_score, "best_goalkeeper")
    if outfield:
        result["mvp"] = _best(lambda s: s.rating + s.goals * 0.5 + s.assists * 0.3, "mvp")

    return {k: v for k, v in result.items() if v}


# ---------------------------------------------------------------------------
# Player Comparison (section 43)
# ---------------------------------------------------------------------------

def compare_players(a: PlayerMatchStat, b: PlayerMatchStat) -> dict:
    """Side-by-side comparison of two players."""
    fields = [
        ("rating", "Рейтинг"),
        ("goals", "Голы"), ("assists", "Ассисты"),
        ("xg", "xG"), ("xa", "xA"),
        ("total_shots", "Удары"), ("shots_on_target", "Удары в створ"),
        ("accurate_passes", "Точные пасы"), ("pass_completion_pct", "Точность паса %"),
        ("key_passes", "Ключевые пасы"),
        ("successful_dribbles", "Дриблинг"), ("touches", "Касания"),
        ("tackles_won", "Отборы"), ("interceptions", "Перехваты"),
        ("defensive_actions", "Защитные действия"),
        ("duels_won", "Выиграно дуэлей"), ("duel_win_pct", "% выиг. дуэлей"),
        ("possession_lost", "Потери"),
    ]
    comparison = []
    for attr, label in fields:
        va = getattr(a, attr) if not callable(getattr(a.__class__, attr, None)) else getattr(a, attr)
        vb = getattr(b, attr) if not callable(getattr(b.__class__, attr, None)) else getattr(b, attr)
        winner = a.player_id if va > vb else (b.player_id if vb > va else "draw")
        comparison.append({
            "metric": attr, "label": label,
            "player_a": va, "player_b": vb, "winner": winner,
        })

    return {
        "player_a": {"player_id": a.player_id, "player_name": a.player_name, "position": a.position},
        "player_b": {"player_id": b.player_id, "player_name": b.player_name, "position": b.position},
        "comparison": comparison,
        "ai_verdict": _comparison_verdict(a, b),
    }


def _comparison_verdict(a: PlayerMatchStat, b: PlayerMatchStat) -> str:
    if a.rating > b.rating + 0.5:
        return f"{a.player_name} провёл значительно лучший матч (рейтинг {a.rating} vs {b.rating})."
    elif b.rating > a.rating + 0.5:
        return f"{b.player_name} провёл значительно лучший матч (рейтинг {b.rating} vs {a.rating})."
    return (f"Игроки показали схожий уровень ({a.player_name} {a.rating} vs "
            f"{b.player_name} {b.rating}). {a.player_name} был лучше в атаке, "
            f"{b.player_name} — в обороне." if a.xg > b.xg else
            f"Схожий уровень: {a.player_name} {a.rating} vs {b.player_name} {b.rating}.")


# ---------------------------------------------------------------------------
# Tab AI summaries (section 47)
# ---------------------------------------------------------------------------

def tab_ai_summary(stats: list[PlayerMatchStat], tab: str) -> str:
    """Generate a short AI insight for a stats tab."""
    if not stats:
        return "Нет данных."

    if tab == "general":
        best = max(stats, key=lambda s: s.rating)
        worst = min(stats, key=lambda s: s.rating)
        hidden = max(stats, key=lambda s: s.recoveries + s.interceptions)
        return (f"Лучший игрок: {best.player_name} ({best.rating}/10). "
                f"Слабое звено: {worst.player_name} ({worst.rating}/10). "
                f"Скрытый вклад: {hidden.player_name} "
                f"({hidden.recoveries} подборов, {hidden.interceptions} перехватов).")

    elif tab == "attacking":
        best_xg = max(stats, key=lambda s: s.xg)
        worst_r = max((s for s in stats if s.total_shots > 0), key=lambda s: s.big_chances_missed, default=stats[0])
        creator = max(stats, key=lambda s: s.xa + s.big_chances_created * 0.3)
        return (f"Главная угроза: {best_xg.player_name} (xG={best_xg.xg:.2f}). "
                f"Не реализовал: {worst_r.player_name} (big chances missed: {worst_r.big_chances_missed}). "
                f"Лучший ассистент: {creator.player_name} (xA={creator.xa:.2f}).")

    elif tab == "defending":
        best_def = max(stats, key=lambda s: s.defensive_reliability_index)
        worst_def = min(stats, key=lambda s: s.zone_protection_score)
        return (f"Лучший в обороне: {best_def.player_name} "
                f"({best_def.tackles_won} отбора, {best_def.interceptions} перехватов). "
                f"Проблемная зона у {worst_def.player_name} "
                f"({worst_def.dribbled_past} раз обошли).")

    elif tab == "passing":
        best_pass = max(stats, key=lambda s: s.build_up_quality_index)
        creator = max(stats, key=lambda s: s.xa + s.key_passes * 0.2)
        worst_press = max(stats, key=lambda s: s.possession_lost)
        return (f"Лучший диспетчер: {best_pass.player_name} "
                f"({best_pass.pass_completion_pct}% точность). "
                f"Создавал моменты: {creator.player_name} (xA={creator.xa:.2f}). "
                f"Больше всего потерь: {worst_press.player_name} ({worst_press.possession_lost}).")

    elif tab == "duels":
        best_duel = max(stats, key=lambda s: s.physical_dominance_index)
        worst_duel = min(stats, key=lambda s: s.duel_win_pct if s.total_duels > 3 else 100)
        return (f"Физически доминировал: {best_duel.player_name} "
                f"({best_duel.duels_won}/{best_duel.total_duels} дуэлей). "
                f"Проигрывал борьбу: {worst_duel.player_name} "
                f"({worst_duel.duel_win_pct}% побед).")

    elif tab == "goalkeeping":
        gks = [s for s in stats if s.position == "GK"]
        if not gks:
            return "Вратарей нет в выборке."
        best_gk = max(gks, key=lambda s: s.goalkeeper_impact_score)
        return (f"{best_gk.player_name}: {best_gk.goalkeeper_saves} сейвов, "
                f"предотвратил {best_gk.goals_prevented:.2f} xG. "
                f"{'Сухой матч. ' if best_gk.goals_conceded == 0 else ''}"
                f"Ключевые сейвы: {best_gk.big_saves}.")

    return "Нет данных."
