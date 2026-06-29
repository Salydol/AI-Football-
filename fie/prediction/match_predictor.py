"""
MatchPredictor — предсказывает исход матча и динамику игры.

Модель: CatBoost (если установлен) или логистическая регрессия (fallback).
Обучение: на StatsBomb данных или любом CSV с фичами + метками.

Три класса исхода:
  0 = LEFT wins  (команда с меньшим X-центром тяжести)
  1 = DRAW
  2 = RIGHT wins

Без обученной модели возвращает rule-based предсказание на основе
территориального преимущества и физической активности.

Использование:
    predictor = MatchPredictor()
    features = extractor.get_features()
    result = predictor.predict(features)
    # PredictionResult(outcome="left_win", probabilities={...}, confidence=0.72)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fie.prediction.features import MatchFeatures


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Результат предсказания исхода матча."""
    outcome: str                        # "left_win" / "draw" / "right_win"
    probabilities: dict[str, float]     # {"left_win": 0.6, "draw": 0.2, "right_win": 0.2}
    confidence: float                   # уверенность модели (макс вероятность)
    method: str                         # "catboost" / "logistic" / "rule_based"
    features_used: dict                 # входные фичи (для отладки)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "features_used": self.features_used,
        }


OUTCOME_NAMES = ["left_win", "draw", "right_win"]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class MatchPredictor:
    """
    Предсказывает исход матча из вектора фич.

    Args:
        model_path: Путь к сохранённой модели (.cbm для CatBoost).
                    Если None или файл не существует — используется rule-based.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model = None
        self._method = "rule_based"

        if model_path and Path(model_path).exists():
            self._load_model(Path(model_path))

    def predict(self, features: MatchFeatures) -> PredictionResult:
        """Предсказать исход по вектору фич."""
        if self._model is not None and self._method == "catboost":
            return self._predict_catboost(features)
        elif self._model is not None and self._method == "logistic":
            return self._predict_logistic(features)
        else:
            return self._predict_rule_based(features)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: str | Path | None = None,
        iterations: int = 500,
    ) -> float:
        """
        Обучить CatBoost модель.

        Args:
            X: (N, n_features) фичи
            y: (N,) метки 0/1/2
            save_path: куда сохранить модель
            iterations: количество деревьев

        Returns:
            Accuracy на тренировочных данных
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return self._train_logistic(X, y, save_path)

        self._model = CatBoostClassifier(
            iterations=iterations,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=42,
            verbose=50,
            class_names=OUTCOME_NAMES,
        )

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self._model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
        )
        self._method = "catboost"

        if save_path:
            self._model.save_model(str(save_path))

        preds = self._model.predict(X_train).flatten()
        return float((preds == y_train).mean())

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    def _predict_catboost(self, features: MatchFeatures) -> PredictionResult:
        x = features.to_array().reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        probs = {name: float(p) for name, p in zip(OUTCOME_NAMES, proba)}
        return PredictionResult(
            outcome=OUTCOME_NAMES[idx],
            probabilities=probs,
            confidence=float(proba[idx]),
            method="catboost",
            features_used=features.to_dict(),
        )

    def _predict_logistic(self, features: MatchFeatures) -> PredictionResult:
        x = features.to_array().reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        probs = {name: float(p) for name, p in zip(OUTCOME_NAMES, proba)}
        return PredictionResult(
            outcome=OUTCOME_NAMES[idx],
            probabilities=probs,
            confidence=float(proba[idx]),
            method="logistic",
            features_used=features.to_dict(),
        )

    def _predict_rule_based(self, features: MatchFeatures) -> PredictionResult:
        """
        Rule-based предсказание когда нет обученной модели.

        Логика:
          - Территориальное преимущество (мяч на чьей половине)
          - Физическая активность
          - Прессинг интенсивность
          - Momentum
        """
        # LEFT advantage score (0..1, >0.5 = left dominates)
        left_score = 0.0
        weights_total = 0.0

        def add(value, weight):
            nonlocal left_score, weights_total
            left_score += value * weight
            weights_total += weight

        # Территория: мяч больше на половине right = left атакует
        add(1.0 - features.left_territory_pct, 0.3)

        # Физика
        left_phys = (features.left_distance_norm + features.left_sprint_count_norm) / 2
        right_phys = (features.right_distance_norm + features.right_sprint_count_norm) / 2
        total_phys = max(left_phys + right_phys, 0.01)
        add(left_phys / total_phys, 0.2)

        # Прессинг
        left_press = features.left_pressing_intensity + features.left_high_press_pct
        right_press = features.right_pressing_intensity + features.right_high_press_pct
        total_press = max(left_press + right_press, 0.01)
        add(left_press / total_press, 0.25)

        # Momentum
        add(features.left_momentum / max(features.left_momentum + features.right_momentum, 0.01), 0.25)

        left_adv = left_score / max(weights_total, 0.01)

        # Конвертировать в вероятности
        draw_prob = max(0.15, 0.35 - abs(left_adv - 0.5) * 0.8)
        remaining = 1.0 - draw_prob

        if left_adv > 0.5:
            left_win_prob = remaining * (0.5 + (left_adv - 0.5) * 1.5)
            right_win_prob = remaining - left_win_prob
        else:
            right_win_prob = remaining * (0.5 + (0.5 - left_adv) * 1.5)
            left_win_prob = remaining - right_win_prob

        left_win_prob = float(np.clip(left_win_prob, 0.05, 0.85))
        right_win_prob = float(np.clip(right_win_prob, 0.05, 0.85))
        draw_prob = float(np.clip(1.0 - left_win_prob - right_win_prob, 0.05, 0.4))

        # Нормализовать
        total = left_win_prob + draw_prob + right_win_prob
        probs = {
            "left_win": left_win_prob / total,
            "draw": draw_prob / total,
            "right_win": right_win_prob / total,
        }

        outcome = max(probs, key=probs.get)
        return PredictionResult(
            outcome=outcome,
            probabilities=probs,
            confidence=probs[outcome],
            method="rule_based",
            features_used=features.to_dict(),
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, path: Path) -> None:
        if path.suffix == ".cbm":
            try:
                from catboost import CatBoostClassifier
                self._model = CatBoostClassifier()
                self._model.load_model(str(path))
                self._method = "catboost"
                return
            except ImportError:
                pass
        if path.suffix in (".pkl", ".joblib"):
            try:
                import joblib
                self._model = joblib.load(str(path))
                self._method = "logistic"
            except Exception:
                pass

    def _train_logistic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: str | Path | None = None,
    ) -> float:
        """Fallback: sklearn LogisticRegression если нет CatBoost."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import joblib

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self._model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self._model.fit(X_scaled, y)
        self._method = "logistic"
        self._scaler = scaler

        if save_path:
            joblib.dump({"model": self._model, "scaler": scaler}, str(save_path))

        preds = self._model.predict(X_scaled)
        return float((preds == y).mean())
