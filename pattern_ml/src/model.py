"""Modele ML: klasyfikacja kierunku (LONG/SHORT/NEUTRALNY) + regresja kwantylowa
przyszłej stopy zwrotu (do narysowania stożka prognozy na wykresie).

Klasyfikator to miękki ensemble (soft-voting) RandomForest + HistGradientBoosting -
połączenie modelu opartego na baggingu z nowoczesnym modelem boostingowym zwykle
daje stabilniejsze i lepiej skalibrowane prawdopodobieństwa niż pojedynczy model,
co jest dziś standardowym podejściem do danych tabelarycznych (m.in. finansowych).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

LABELS = {1: "LONG", -1: "SHORT", 0: "NEUTRALNY"}
QUANTILES = (0.1, 0.5, 0.9)


def make_classifier() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
    return VotingClassifier(estimators=[("rf", rf), ("hgb", hgb)], voting="soft")


@dataclass
class TrainResult:
    model: VotingClassifier
    report: str
    feature_importances: pd.Series
    cv_accuracy: float


def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = 0) -> TrainResult:
    """Trenuje ensemble RF + HistGradientBoosting z walidacją krzyżową szeregu
    czasowego (bez przecieku danych z przyszłości).

    `gap` (tzw. purging) pomija `gap` próbek między train a test foldem - istotne
    bo etykieta każdej próbki zależy od ceny `horizon` świec w przód, więc bez
    tej przerwy ostatnie próbki treningowe "widziałyby" fragment danych z okna
    testowego (przeciek informacji z przyszłości na granicy foldów).
    """
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 50)), gap=gap)

    accuracies = []
    last_pred, last_true = None, None
    for train_idx, test_idx in tscv.split(X):
        clf = make_classifier()
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = clf.predict(X.iloc[test_idx])
        accuracies.append((pred == y.iloc[test_idx].values).mean())
        last_pred, last_true = pred, y.iloc[test_idx]

    report = classification_report(
        last_true, last_pred, labels=[-1, 0, 1],
        target_names=["SHORT", "NEUTRALNY", "LONG"], zero_division=0,
    )

    # finalny model trenowany na wszystkich dostępnych danych historycznych
    final_model = make_classifier()
    final_model.fit(X, y)

    rf_importances = pd.Series(
        final_model.named_estimators_["rf"].feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return TrainResult(
        model=final_model,
        report=report,
        feature_importances=rf_importances,
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


def train_quantile_models(X: pd.DataFrame, y_reg: pd.Series) -> dict:
    """Trenuje regresory kwantylowe (10/50/90 percentyl) przyszłej stopy zwrotu.

    Pozwalają narysować na wykresie prognozowany "stożek" ceny (dolna/mediana/górna
    granica), zamiast pojedynczej deterministycznej liczby - uczciwiej oddaje
    niepewność prognozy rynkowej.
    """
    models = {}
    for q in QUANTILES:
        reg = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
        reg.fit(X, y_reg)
        models[q] = reg
    return models


def predict_price_path(quantile_models: dict, features: pd.DataFrame, feature_cols: list, last_close: float):
    """Zwraca prognozowane ceny (dict kwantyl->cena) za horyzont modelu, na bazie ostatniej świecy."""
    latest = features.dropna(subset=feature_cols).iloc[[-1]]
    X_latest = latest[feature_cols].astype(float)

    prices = {}
    for q, model in quantile_models.items():
        predicted_return = model.predict(X_latest)[0]
        prices[q] = last_close * (1 + predicted_return)

    # monotoniczność kwantyli (regresory trenowane niezależnie mogą się lekko "przecinać")
    ordered = sorted(prices.items())
    values = np.sort([v for _, v in ordered])
    return {q: v for (q, _), v in zip(ordered, values)}
