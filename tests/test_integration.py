"""
End-to-end integration test for Football Intelligence Engine v0.9.0

Tests the full analytics pipeline using mock data (no GPU, no real video).

Run with:
  cd fie
  pytest tests/test_integration.py -v
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import types
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import pytest


# ===========================================================================
# Module-level mocks (applied before any fie.* import)
# ===========================================================================

def _setup_mocks():
    # pydantic_settings
    ps = types.ModuleType("pydantic_settings")
    class _BS:
        def __init__(self, **kw): pass
    ps.BaseSettings = _BS
    ps.SettingsConfigDict = lambda **kw: kw
    sys.modules.setdefault("pydantic_settings", ps)

    # ultralytics
    um = types.ModuleType("ultralytics")
    um.YOLO = type("YOLO", (), {"__init__": lambda s, *a, **k: None})
    sys.modules.setdefault("ultralytics", um)

    # Tracking dataclasses used by analytics modules
    @dataclass(slots=True)
    class _TrackedPlayer:
        player_id: int; x: float; y: float
        speed: float = 0.0; acceleration: float = 0.0; direction: float = 0.0

    @dataclass(slots=True)
    class _TrackedBall:
        x: float = 52.5; y: float = 34.0; confidence: float = 1.0

    @dataclass
    class _TrackingFrame:
        frame_idx: int = 0; timestamp: float = 0.0
        players: list = dc_field(default_factory=list)
        ball: object = None
        def to_dict(self): return {"frame_idx": self.frame_idx}

    tp = types.ModuleType("fie.tracking.pipeline")
    tp.TrackedPlayer = _TrackedPlayer
    tp.TrackedBall   = _TrackedBall
    tp.TrackingFrame = _TrackingFrame
    sys.modules["fie.tracking.pipeline"] = tp

    for m in ["fie.tracking", "fie.tracking.detector", "fie.tracking.calibration",
              "fie.tracking.source", "fie.tracking.metrics",
              "fie.tactical", "fie.tactical.pressing", "fie.tactical.compactness"]:
        sys.modules.setdefault(m, types.ModuleType(m))

    class _PA:
        def __init__(self, *a, **k): pass
        def analyze(self, *a, **k): return types.SimpleNamespace(intensity=0.6, line=0.55)
    class _CA:
        def __init__(self, *a, **k): pass
        def analyze(self, *a, **k): return types.SimpleNamespace(compactness=0.5, width=0.7)
    sys.modules["fie.tactical.pressing"].PressingAnalyzer = _PA
    sys.modules["fie.tactical.compactness"].CompactnessAnalyzer = _CA

    cfg = types.ModuleType("fie.config")
    cfg.settings = types.SimpleNamespace(
        field_length_m=105.0, field_width_m=68.0, yolo_device="cuda")
    sys.modules["fie.config"] = cfg

    for mod in ["torch", "torch.nn", "torch.utils", "torch.utils.data",
                "lightning", "lightning.pytorch", "cv2", "supervision",
                "scipy", "scipy.signal", "catboost", "anthropic", "openai",
                "reportlab", "matplotlib", "matplotlib.pyplot", "matplotlib.patches"]:
        sys.modules.setdefault(mod, types.ModuleType(mod))

    # numpy with linalg
    nm = types.ModuleType("numpy")
    la = types.ModuleType("numpy.linalg")

    class _Arr(list):
        def __sub__(self, o): return _Arr(a - b for a, b in zip(self, o))
        def __rsub__(self, o): return _Arr(b - a for a, b in zip(self, o))

    nm.ndarray = type("ndarray", (), {})
    nm.array   = lambda a, **k: _Arr(a) if hasattr(a, "__iter__") else _Arr([a])
    nm.mean    = lambda a, **k: sum(a) / len(a) if len(a) else 0.0
    nm.std     = lambda a, **k: 0.0
    nm.sqrt    = math.sqrt
    nm.abs     = abs
    nm.clip    = lambda v, lo, hi: max(lo, min(hi, v))
    la.norm    = lambda x: math.sqrt(sum(v ** 2 for v in x))
    nm.linalg  = la
    sys.modules["numpy"]        = nm
    sys.modules["numpy.linalg"] = la

    # loguru
    lm = types.ModuleType("loguru")
    class _L:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def debug(self, *a, **k): pass
        def error(self, *a, **k): pass
    lm.logger = _L()
    sys.modules["loguru"] = lm

    # expose TrackingFrame for tests
    return tp.TrackingFrame, tp.TrackedPlayer


_TrackingFrame, _TrackedPlayer = _setup_mocks()


# ===========================================================================
# 1. Fatigue Analyzer
# ===========================================================================

class TestFatigueAnalyzer:
    def test_basic_fatigue_and_level(self):
        from fie.analytics.fatigue import FatigueAnalyzer, FatigueLevel
        a = FatigueAnalyzer(fps=25.0)
        for i in range(375):
            a.update(_TrackingFrame(
                frame_idx=i, timestamp=i / 25.0,
                players=[_TrackedPlayer(1, float(i) * 0.5, 30.0,
                         speed=5.0 + (i % 50) * 0.5, acceleration=0.2, direction=0.0)],
            ))
        s = a.get_player_state(1)
        assert s is not None, "get_player_state returned None"
        assert 0 <= s.fatigue_score <= 100
        assert s.fatigue_level in list(FatigueLevel)

    def test_unknown_player_returns_none(self):
        from fie.analytics.fatigue import FatigueAnalyzer
        assert FatigueAnalyzer().get_player_state(999) is None

    def test_multiple_players(self):
        from fie.analytics.fatigue import FatigueAnalyzer
        a = FatigueAnalyzer(fps=25.0)
        for pid in [1, 2, 3]:
            for i in range(50):
                a.update(_TrackingFrame(
                    frame_idx=i, timestamp=i / 25.0,
                    players=[_TrackedPlayer(pid, float(i) * 0.3, 30.0 + pid)],
                ))
        for pid in [1, 2, 3]:
            assert a.get_player_state(pid) is not None


# ===========================================================================
# 2. Player Passport
# ===========================================================================

class TestPlayerPassport:
    @staticmethod
    def _make_record(i: int):
        from fie.analytics.passport import MatchRecord
        return MatchRecord(
            match_id=f"m{i:03d}", date=f"2024-{i + 1:02d}-10",
            opponent="Opponent FC", duration_minutes=90.0,
            distance_km=10.5 + i * 0.2, max_speed_kmh=30.0 + i * 0.3,
            avg_speed_kmh=8.5, sprint_count=18 + i, high_accel_count=25 + i,
            physical_rating=7.0 + i * 0.1, tactical_rating=6.8,
            overall_rating=6.9 + i * 0.1,
            avg_x=52.0, avg_y=34.0, time_near_ball_s=1200.0,
            fatigue_score=45.0 + i, injury_risk=0.2,
        )

    def test_save_load(self, tmp_path):
        from fie.analytics.passport import PlayerPassport
        pp = PlayerPassport.load_or_create(7, "CM", str(tmp_path))
        for i in range(5):
            pp.add_match(self._make_record(i))
        pp.save()
        loaded = PlayerPassport.load_or_create(7, "CM", str(tmp_path))
        assert len(loaded._matches) == 5

    def test_profile_trend(self, tmp_path):
        from fie.analytics.passport import PlayerPassport
        pp = PlayerPassport.load_or_create(10, "ST", str(tmp_path))
        for i in range(5):
            pp.add_match(self._make_record(i))
        p = pp.get_profile()
        assert p is not None
        assert 0 <= p.progress_pct <= 100
        assert p.trend in ("improving", "stable", "declining", "insufficient_data")


# ===========================================================================
# 3. Team DNA
# ===========================================================================

class TestTeamDNA:
    @staticmethod
    def _make_dna(team_id: str, save_dir, pi: float):
        from fie.analytics.team_dna import TeamDNA, TeamDNAVector
        d = TeamDNA.load_or_create(team_id, str(save_dir))
        d.add_match_dna(TeamDNAVector(
            pressing_intensity=pi, pressing_line=0.5, tempo=0.5,
            territory=0.5, defensive_line=0.5, attack_width=0.5,
            aggression=pi, compactness=0.5,
        ))
        d.save()
        return TeamDNA.load_or_create(team_id, str(save_dir))

    def test_profile(self, tmp_path):
        d = self._make_dna("home", tmp_path, 0.75)
        p = d.get_profile()
        assert p is not None
        assert 0.0 <= p.dna.pressing_intensity <= 1.0

    def test_similarity(self, tmp_path):
        high = self._make_dna("high_press", tmp_path, 0.90)
        low  = self._make_dna("low_press",  tmp_path, 0.10)
        r_same = high.compare(high)
        r_diff = high.compare(low)
        assert "similarity" in r_same
        assert r_same["similarity"] >= r_diff["similarity"]
        assert 0.0 <= r_diff["similarity"] <= 1.0


# ===========================================================================
# 4. Academy Progress Tracker
# ===========================================================================

class TestAcademyTracker:
    @staticmethod
    def _add_sessions(tracker, n: int, age: int = 17):
        from fie.analytics.academy import AcademySession
        for i in range(n):
            tracker.add_session(AcademySession(
                session_id=f"s{i:03d}", date=f"2024-{i + 1:02d}-10",
                session_type="match", age_at_session=age,
                distance_km=9.5 + i * 0.1, max_speed_kmh=28.5 + i * 0.2,
                sprint_count=12 + i, high_accel_count=8,
                physical_rating=6.5 + i * 0.1, tactical_rating=6.0,
                overall_rating=6.3 + i * 0.1, coach_note="",
            ))

    def test_save_load(self, tmp_path):
        from fie.analytics.academy import AcademyTracker
        t = AcademyTracker.load_or_create(99, "Ivan", "WNG", str(tmp_path))
        self._add_sessions(t, 6)
        t.save()
        loaded = AcademyTracker.load(tmp_path / "player_99.json")
        assert len(loaded._sessions) == 6

    def test_profile(self, tmp_path):
        from fie.analytics.academy import AcademyTracker, AgeGroup
        t = AcademyTracker.load_or_create(99, "Ivan", "WNG", str(tmp_path))
        self._add_sessions(t, 6)
        p = t.get_profile()
        assert p is not None
        assert p.age_group in [g.value for g in AgeGroup]
        assert 0 <= p.development_score <= 100

    def test_readiness(self, tmp_path):
        from fie.analytics.academy import AcademyTracker
        t = AcademyTracker.load_or_create(77, "Test", "CM", str(tmp_path))
        self._add_sessions(t, 8, age=20)
        p = t.get_profile()
        r = p.first_team_readiness
        assert r.readiness_label in ("not_ready", "developing", "close", "ready")
        assert 0 <= r.readiness_pct <= 100
        assert isinstance(r.recommendations, list)


# ===========================================================================
# 5. Video Index + Query Engine
# ===========================================================================

class TestVideoSearch:
    @staticmethod
    def _make_index():
        from fie.search.video_index import VideoIndex
        idx = VideoIndex(video_id="test_match", fps=25.0, total_frames=135000)
        for frame, etype, conf in [
            (3000, "shot",    0.88), (6000,  "tackle",  0.74),
            (9000, "shot",    0.92), (18000, "dribble", 0.68),
            (27000, "pass",   0.60), (67500, "shot",    0.88),
            (76500, "tackle", 0.75), (90000, "shot",    0.93),
            (121500, "dribble", 0.70), (130500, "shot", 0.86),
        ]:
            idx.add_event(frame, etype, conf, player_id=9)
        return idx

    def test_stats(self):
        idx = self._make_index()
        s = idx.stats()
        assert s["total_events"] == 10
        assert s["event_types"]["shot"] == 5

    def test_save_load(self, tmp_path):
        from fie.search.video_index import VideoIndex
        idx = self._make_index()
        loaded = VideoIndex.load(idx.save(tmp_path))
        assert len(loaded.events) == len(idx.events)

    def test_structured_filters(self):
        idx = self._make_index()
        assert all(e.half == 1 for e in idx.search(half=1))
        assert all(e.half == 2 for e in idx.search(half=2))
        assert all(e.event_type == "shot" for e in idx.search(event_types=["shot"]))

    def test_query_engine(self):
        from fie.search.query_engine import QueryEngine
        idx = self._make_index()
        engine = QueryEngine()
        assert engine._backend == "rule_based"

        r = engine.search("все удары", idx)
        assert all(e.event_type == "shot" for e in r.matched_events)

        r2 = engine.search("first half shots", idx)
        assert all(e.half == 1 and e.event_type == "shot" for e in r2.matched_events)

        r3 = engine.search("опасные моменты второго тайма", idx)
        assert all(e.half == 2 for e in r3.matched_events)

        r4 = engine.search("last 10 minutes", idx)
        assert all(e.minute >= 80 for e in r4.matched_events)

        assert json.dumps(r.to_dict())   # must be JSON-serialisable


# ===========================================================================
# 6. LLM Coach Assistant (rule-based)
# ===========================================================================

class TestCoachAssistant:
    SUMMARY = {
        "teams": "Home vs Away", "score": "1-2", "duration_min": 90,
        "physical": {
            "home": {"distance_km": 108, "sprint_count": 145, "max_speed": 32.1},
            "away": {"distance_km": 112, "sprint_count": 162, "max_speed": 33.5},
        },
        "tactical": {"pressing_intensity": 0.62, "territory_pct": 44.0},
        "mistakes": [{"type": "defensive_gap", "severity": "high", "player_id": 5}],
        "player_ratings": [
            {"player_id": 9, "overall": 8.2, "physical": 8.5, "tactical": 7.9},
            {"player_id": 3, "overall": 5.1, "physical": 4.8, "tactical": 5.4},
        ],
        "fatigue": {"critical_players": [3], "high_risk_players": [7], "team_fatigue_avg": 68},
    }

    def test_rule_based_backend(self):
        from fie.llm.coach_assistant import CoachAssistant
        assert "Rule" in CoachAssistant(force_backend="rule_based").backend_name

    def test_question_answer(self):
        from fie.llm.coach_assistant import CoachAssistant
        a = CoachAssistant(force_backend="rule_based")
        r = a.ask("Почему проблемы с прессингом?", self.SUMMARY)
        assert r.answer and len(r.answer) > 20

    def test_substitution_advice(self):
        from fie.llm.coach_assistant import CoachAssistant
        r = CoachAssistant(force_backend="rule_based").substitution_advice(self.SUMMARY)
        assert r.answer and len(r.answer) > 20

    def test_tactical_report(self):
        from fie.llm.coach_assistant import CoachAssistant
        r = CoachAssistant(force_backend="rule_based").tactical_report(self.SUMMARY)
        assert r.answer and len(r.answer) > 50


# ===========================================================================
# 7. Match Story Generator
# ===========================================================================

class TestMatchStory:
    SUMMARY = {
        "teams": "Spartak vs Zenit", "score": "2-1", "duration_min": 90,
        "physical": {"home": {"distance_km": 111}, "away": {"distance_km": 109}},
        "player_ratings": [{"player_id": 10, "overall": 8.5, "physical": 8.8, "tactical": 8.2}],
        "mistakes": [],
    }

    def test_russian(self):
        from fie.llm.match_story import MatchStoryGenerator
        s = MatchStoryGenerator().generate(self.SUMMARY, "ru", "Spartak", "Zenit")
        assert s.text and len(s.text) > 100 and s.language == "ru"

    def test_english(self):
        from fie.llm.match_story import MatchStoryGenerator
        s = MatchStoryGenerator().generate(self.SUMMARY, "en", "Spartak", "Zenit")
        assert s.text and len(s.text) > 100 and s.language == "en"


# ===========================================================================
# 8. Full pipeline smoke test
# ===========================================================================

class TestFullPipeline:
    def test_analytics_chain(self, tmp_path):
        """Fatigue -> Passport -> TeamDNA -> Academy: data flows, saves/loads."""
        from fie.analytics.fatigue import FatigueAnalyzer
        from fie.analytics.passport import PlayerPassport, MatchRecord
        from fie.analytics.team_dna import TeamDNA, TeamDNAVector
        from fie.analytics.academy import AcademyTracker, AcademySession

        # Fatigue
        fa = FatigueAnalyzer(fps=25.0)
        for i in range(200):
            fa.update(_TrackingFrame(
                frame_idx=i, timestamp=i / 25.0,
                players=[_TrackedPlayer(1, float(i) * 0.4, 30.0,
                         speed=4.0 + (i % 30) * 0.3, acceleration=0.15, direction=0.0)],
            ))
        state = fa.get_player_state(1)
        assert state is not None

        # Passport (fed with fatigue metrics)
        pp = PlayerPassport.load_or_create(1, "CM", str(tmp_path / "pp"))
        pp.add_match(MatchRecord(
            match_id="chain_m001", date="2024-11-10",
            opponent="Test FC", duration_minutes=90.0,
            distance_km=10.5,
            max_speed_kmh=29.0,
            avg_speed_kmh=8.5,
            sprint_count=state.sprint_load or 15, high_accel_count=20,
            physical_rating=7.2, tactical_rating=6.9, overall_rating=7.0,
            avg_x=52.0, avg_y=34.0, time_near_ball_s=1200.0,
            fatigue_score=state.fatigue_score, injury_risk=0.15,
        ))
        pp.save()
        assert pp.get_profile() is not None

        # Team DNA
        dna = TeamDNA.load_or_create("team1", str(tmp_path / "dna"))
        dna.add_match_dna(TeamDNAVector(
            pressing_intensity=0.72, pressing_line=0.60, tempo=0.65,
            territory=0.55, defensive_line=0.50, attack_width=0.75,
            aggression=0.68, compactness=0.52,
        ))
        dna.save()
        p = dna.get_profile()
        assert p is not None
        assert isinstance(dna.compare(dna), dict)

        # Academy
        t = AcademyTracker.load_or_create(1, "Player", "CM", str(tmp_path / "ac"))
        for i in range(4):
            t.add_session(AcademySession(
                session_id=f"s{i}", date=f"2024-{i+1:02d}-15",
                session_type="match", age_at_session=19,
                distance_km=10.5, max_speed_kmh=29.5, sprint_count=16,
                high_accel_count=9, physical_rating=7.0, tactical_rating=6.8,
                overall_rating=6.9, coach_note="",
            ))
        assert t.get_profile() is not None

    def test_search_chain(self, tmp_path):
        """VideoIndex -> save -> load -> QueryEngine search -> JSON output."""
        from fie.search.video_index import VideoIndex
        from fie.search.query_engine import QueryEngine

        idx = VideoIndex(video_id="chain_test", fps=25.0, total_frames=135000)
        for f, e, c in [(3000,"shot",0.88),(9000,"shot",0.92),(75000,"dribble",0.70),(120000,"shot",0.85)]:
            idx.add_event(f, e, c, player_id=9)

        loaded = VideoIndex.load(idx.save(tmp_path))
        engine = QueryEngine()
        r = engine.search("все удары в первом тайме", loaded)
        assert all(e.event_type == "shot" and e.half == 1 for e in r.matched_events)
        json.dumps(r.to_dict())   # must not raise

    def test_llm_chain(self):
        """CoachAssistant -> MatchStory: full text output, JSON-serialisable."""
        from fie.llm.coach_assistant import CoachAssistant
        from fie.llm.match_story import MatchStoryGenerator

        summary = {
            "teams": "A vs B", "score": "2-0", "duration_min": 90,
            "physical": {"home": {"distance_km": 110, "sprint_count": 150},
                         "away": {"distance_km": 105, "sprint_count": 130}},
            "player_ratings": [{"player_id": 7, "overall": 8.1, "physical": 8.0, "tactical": 8.2}],
            "mistakes": [],
            "fatigue": {"critical_players": [], "high_risk_players": [3], "team_fatigue_avg": 55},
        }
        a = CoachAssistant(force_backend="rule_based")
        q = a.ask("Кого заменить?", summary)
        story = MatchStoryGenerator(assistant=a).generate(summary, "ru", "A", "B")
        result = {"answer": q.answer, "story": story.text, "backend": story.backend}
        json.dumps(result)   # must not raise


# ===========================================================================
# 9. Opponent Weakness Scanner
# ===========================================================================

class TestOpponentScanner:
    @staticmethod
    def _make_match(i: int):
        from fie.analytics.opponent import OpponentMatch, ZoneStats
        return OpponentMatch(
            match_id=f"m{i}", date=f"2024-{i+1:02d}-10",
            opponent_name="Test FC",
            distance_km=108.0, sprint_count=140 + i, max_speed_kmh=33.0,
            high_accel_count=95,
            pressing_intensity=0.72, pressing_line=0.60,
            compactness=0.30, territory=0.55,
            goals_scored=1, goals_conceded=2,
            xg_for=1.2, xg_against=2.1,
            speed_by_period={
                "0-15": 8.2, "15-30": 7.9, "30-45": 7.5,
                "45-60": 7.1, "60-75": 6.6, "75-90": 6.0,
            },
            zone_stats=[
                ZoneStats("right_def", losses=4, mistakes=3, xg_conceded=0.9, pressing_failures=2),
                ZoneStats("center_mid", losses=2, mistakes=1, xg_conceded=0.3),
            ],
            weak_players={5: "позиционные ошибки", 3: "медленный возврат"},
        )

    def test_basic_scan(self):
        from fie.analytics.opponent import OpponentScanner
        scanner = OpponentScanner("Real Madrid")
        for i in range(3):
            scanner.add_match(self._make_match(i))
        report = scanner.analyze()
        assert report.matches_analyzed == 3
        assert 0 <= report.overall_vulnerability_score <= 100
        assert isinstance(report.attack_recommendations, list)
        assert len(report.attack_recommendations) > 0

    def test_weak_zones(self):
        from fie.analytics.opponent import OpponentScanner
        scanner = OpponentScanner("FC Test")
        for i in range(3):
            scanner.add_match(self._make_match(i))
        report = scanner.analyze()
        assert len(report.weak_zones) > 0
        assert all(0 <= z.weakness_score <= 1 for z in report.weak_zones)

    def test_fatigue_windows(self):
        from fie.analytics.opponent import OpponentScanner
        scanner = OpponentScanner("FC Tired")
        for i in range(3):
            scanner.add_match(self._make_match(i))
        report = scanner.analyze()
        # speed_by_period has a big drop in 75-90 -> should detect fatigue window
        assert len(report.fatigue_windows) > 0
        assert all(fw.avg_speed_drop_pct > 0 for fw in report.fatigue_windows)

    def test_tactical_vulnerabilities(self):
        from fie.analytics.opponent import OpponentScanner
        scanner = OpponentScanner("FC Press")
        for i in range(3):
            scanner.add_match(self._make_match(i))
        report = scanner.analyze()
        # pressing_intensity=0.72 → should detect high_press_counter_vulnerable
        types = [v.type for v in report.tactical_vulnerabilities]
        assert "high_press_counter_vulnerable" in types

    def test_save_load(self, tmp_path):
        from fie.analytics.opponent import OpponentScanner, OpponentWeaknessReport
        scanner = OpponentScanner("Saved FC")
        for i in range(2):
            scanner.add_match(self._make_match(i))
        report = scanner.analyze()
        path = report.save(tmp_path)
        loaded = OpponentWeaknessReport.load(path)
        assert loaded.opponent_name == "Saved FC"
        assert loaded.matches_analyzed == 2
        assert json.dumps(loaded.to_dict())  # JSON-serialisable


# ===========================================================================
# 10. Explainable AI
# ===========================================================================

class TestExplainableAI:
    def test_goal_prob_high(self):
        from fie.explainability.explainer import Explainer
        e = Explainer("ru").explain_goal_probability(
            0.72, compactness=0.25, distance_to_goal=9.0,
            players_between=0, shot_angle_deg=50.0, under_pressure=False,
        )
        assert e.severity == "critical"
        assert len(e.factors) > 0
        assert "72" in e.verdict or "72.0" in e.verdict
        assert json.dumps(e.to_dict())

    def test_goal_prob_low(self):
        from fie.explainability.explainer import Explainer
        e = Explainer("en").explain_goal_probability(
            0.08, compactness=0.75, distance_to_goal=32.0,
            players_between=4, shot_angle_deg=8.0,
        )
        assert e.severity == "info"
        assert e.language == "en"

    def test_fatigue_critical(self):
        from fie.explainability.explainer import Explainer
        e = Explainer("ru").explain_fatigue(
            82.0, player_id=9, sprint_count=48,
            speed_drop_pct=22.0, minutes_played=78.0,
        )
        assert e.severity == "critical"
        assert "#9" in e.verdict
        assert len(e.factors) >= 2

    def test_tactical_mistake_defensive_gap(self):
        from fie.explainability.explainer import Explainer
        e = Explainer("ru").explain_tactical_mistake(
            "defensive_gap", player_id=5, severity="high",
            zone="right_def", minute=67,
        )
        assert e.severity == "critical"
        assert "#5" in e.verdict
        assert len(e.factors) >= 2

    def test_match_prediction(self):
        from fie.explainability.explainer import Explainer
        e = Explainer("en").explain_match_prediction(
            0.65, 0.20, 0.15,
            home_team="City", away_team="United",
            home_xg=1.8, away_xg=0.9,
        )
        assert "City" in e.verdict
        assert e.confidence > 0
        assert json.dumps(e.to_dict())

    def test_team_dna_explanation(self):
        from fie.explainability.explainer import Explainer
        dna = {"pressing_intensity": 0.82, "pressing_line": 0.70,
               "tempo": 0.75, "territory": 0.65,
               "attack_width": 0.60, "aggression": 0.78, "compactness": 0.55}
        e = Explainer("ru").explain_team_dna(dna, team_name="Спартак")
        assert "Спартак" in e.verdict
        assert len(e.factors) > 0

    def test_scouting_explanation(self):
        from fie.explainability.explainer import Explainer
        e = Explainer("ru").explain_scouting(
            85.0, player_id=17, position="WNG",
            strengths=["высокая скорость", "дриблинг"],
            weaknesses=["игра без мяча"],
            readiness_pct=78.0,
        )
        assert e.severity == "critical"
        assert "#17" in e.verdict
        assert "78" in " ".join(e.factors)

    def test_to_text(self):
        from fie.explainability.explainer import Explainer
        e = Explainer("en").explain_fatigue(60.0, player_id=3, sprint_count=38)
        text = e.to_text()
        assert "1." in text   # numbered factors
