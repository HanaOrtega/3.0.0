"""Model ML (RandomForest) do przewidywania kierunku ceny (LONG/SHORT/NEUTRALNY)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

LABELS = {1: "LONG", -1: "SHORT", 0: "NEUTRALNY"}


@dataclass
class TrainResult:
    model: RandomForestClassifier
    report: str
    feature_importances: pd.Series
    cv_accuracy: float


def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> TrainResult:
    """Trenuje RandomForest z walidacją krzyżową szeregu czasowego (bez przecieku danych z przyszłości)."""
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 50)))

    accuracies = []
    last_pred, last_true = None, None
    for train_idx, test_idx in tscv.split(X):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = clf.predict(X.iloc[test_idx])
        accuracies.append((pred == y.iloc[test_idx].values).mean())
        last_pred, last_true = pred, y.iloc[test_idx]

    report = classification_report(
        last_true, last_pred, labels=[-1, 0, 1],
        target_names=["SHORT", "NEUTRALNY", "LONG"], zero_division=0,
    )

    # finalny model trenowany na wszystkich dostępnych danych historycznych
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    importances = pd.Series(
        final_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return TrainResult(
        model=final_model,
        report=report,
        feature_importances=importances,
        cv_accuracy=float(np.mean(accuracies)),
    )


def predict_latest(result: TrainResult, features: pd.DataFrame, feature_cols: list):
    """Zwraca (etykieta_tekstowa, prawdopodobienstwa_dict) dla najnowszej dostępnej świecy."""
    latest = features.dropna(subset=feature_cols).iloc[[-1]]
    X_latest = latest[feature_cols].astype(float)

    pred = result.model.predict(X_latest)[0]
    proba = result.model.predict_proba(X_latest)[0]
    proba_dict = {LABELS[c]: float(p) for c, p in zip(result.model.classes_, proba)}

    return LABELS[pred], proba_dict, latest.index[-1]
