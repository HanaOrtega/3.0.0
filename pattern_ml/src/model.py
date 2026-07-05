"""Modele ML: klasyfikacja kierunku (LONG/SHORT/NEUTRALNY) + regresja kwantylowa
przyszłej stopy zwrotu (do narysowania stożka prognozy na wykresie).

Klasyfikator to stacking ensemble (RandomForest + HistGradientBoosting, meta-model
LogisticRegression) - zamiast prostego uśredniania (soft-voting) meta-model uczy
się, jak ważyć predykcje obu modeli bazowych na podstawie ich (out-of-fold)
trafności, co zwykle daje lepiej skalibrowany wynik końcowy niż stałe wagi.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

LABELS = {1: "LONG", -1: "SHORT", 0: "NEUTRALNY"}
QUANTILES = (0.1, 0.5, 0.9)

MIN_TRAIN_FOLD = 60
MIN_TEST_FOLD = 20


def make_classifier() -> StackingClassifier:
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
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    # cv=3 (StratifiedKFold domyślny dla klasyfikacji): sklearn generuje przez to
    # out-of-fold predykcje bazowych modeli do treningu meta-modelu za pomocą
    # cross_val_predict, który wymaga PEŁNEJ partycji danych - TimeSeriesSplit
    # tego nie zapewnia (pierwsza porcja próbek nigdy nie trafia do żadnego test
    # folda), więc nie da się jej tu podstawić. To niewielkie ustępstwo dotyczy
    # WYŁĄCZNIE wag łączących oba modele bazowe - nie ma wpływu na uczciwość
    # głównej walidacji walk-forward w train_model (purged_splits poniżej),
    # która pozostaje w pełni respektująca porządek czasowy.
    return StackingClassifier(
        estimators=[("rf", rf), ("hgb", hgb)],
        final_estimator=meta,
        cv=3,
        stack_method="predict_proba",
    )


def choose_n_splits(n_samples: int, max_splits: int = 5) -> int:
    """Dobiera liczbę foldów CV tak, żeby każdy fold miał sensowną liczbę próbek
    (zamiast stałego podziału, który przy krótkiej historii może dać foldy zbyt
    małe do wytrenowania/oceny modelu)."""
    for n_splits in range(max_splits, 1, -1):
        test_size = n_samples // (n_splits + 1)
        train_size = n_samples - n_splits * test_size
        if test_size >= MIN_TEST_FOLD and train_size >= MIN_TRAIN_FOLD:
            return n_splits
    return 2


def purged_splits(n_samples: int, n_splits: int, gap: int = 0, embargo: int = 0):
    """TimeSeriesSplit z purgingiem (`gap`, natywny parametr sklearn) i embargiem
    dodatkowym buforem `embargo` próbek odciętym z KOŃCA train foldu, oprócz `gap`.

    Embargo jest istotne, gdy cechy mają dłuższe okno "pamięci" niż `horizon`
    etykiety (np. SMA50 vs horizon=5) - sam `gap` czyści tylko tyle, ile wynika
    z konstrukcji etykiety, a embargo dodatkowo zabezpiecza przed przeciekiem
    przez autokorelację wskaźników o dłuższym oknie."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for train_idx, test_idx in tscv.split(np.arange(n_samples)):
        if embargo and len(train_idx):
            cutoff = train_idx[-1] - embargo
            train_idx = train_idx[train_idx <= cutoff]
        yield train_idx, test_idx


@dataclass
class TrainResult:
    model: StackingClassifier
    report: str
    feature_importances: pd.Series
    cv_accuracy: float


def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = 0, embargo: int = 0) -> TrainResult:
    """Trenuje stacking ensemble RF + HistGradientBoosting z walidacją krzyżową
    szeregu czasowego (purged + embargo, bez przecieku danych z przyszłości).

    `gap` pomija próbki między train a test foldem odpowiadające horyzontowi
    etykiety; `embargo` dokłada dodatkowy bufor na końcu train foldu, żeby
    długoterminowe wskaźniki (np. SMA50) też nie "widziały" fragmentu test foldu.
    """
    n_splits = min(n_splits, choose_n_splits(len(X)))

    accuracies = []
    last_pred, last_true = None, None
    for train_idx, test_idx in purged_splits(len(X), n_splits, gap=gap, embargo=embargo):
        if len(train_idx) < MIN_TRAIN_FOLD // 2 or len(test_idx) == 0:
            continue
        clf = make_classifier()
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = clf.predict(X.iloc[test_idx])
        accuracies.append((pred == y.iloc[test_idx].values).mean())
        last_pred, last_true = pred, y.iloc[test_idx]

    report = classification_report(
        last_true, last_pred, labels=[-1, 0, 1],
        target_names=["SHORT", "NEUTRALNY", "LONG"], zero_division=0,
    ) if last_true is not None else "Za mało danych na sensowną walidację krzyżową."

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
        cv_accuracy=float(np.mean(accuracies)) if accuracies else float("nan"),
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
