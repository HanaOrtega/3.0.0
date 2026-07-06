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
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

LABELS = {1: "LONG", -1: "SHORT", 0: "NEUTRALNY"}
CANONICAL_CLASSES = np.array([-1, 0, 1])
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


def _reindex_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Ujednolica macierz prawdopodobieństw do stałego układu kolumn [-1, 0, 1],
    wypełniając zerem klasy nieobecne w danym foldzie (rzadki przypadek przy
    bardzo małych/niezbalansowanych foldach) - potrzebne, żeby móc bezpiecznie
    scalać (pool) macierze z różnych foldów przed liczeniem AUC-ROC."""
    out = np.zeros((proba.shape[0], len(CANONICAL_CLASSES)))
    for i, c in enumerate(classes):
        out[:, np.where(CANONICAL_CLASSES == c)[0][0]] = proba[:, i]
    return out


DEFAULT_CONFIDENCE_GRID = np.arange(0.36, 0.76, 0.02)


def _tune_confidence_threshold(
    records: np.ndarray, thresholds=DEFAULT_CONFIDENCE_GRID, min_signal_rate: float = 0.05
):
    """Dobiera próg pewności maksymalizujący PRECYZJĘ sygnałów kierunkowych
    (LONG/SHORT), nie ogólną trafność (accuracy/F1) - bo celem jest ograniczenie
    strat na fałszywych sygnałach, a nie łapanie każdej okazji. Model może więc
    świadomie "milczeć" (NEUTRALNY) częściej, jeśli to podnosi jakość tych
    sygnałów, na które się faktycznie decyduje.

    `records`: tablica (n, 3) kolumn (pewność, klasa_przewidziana, klasa_prawdziwa)
    zebrana z out-of-fold predykcji walidacji krzyżowej. `min_signal_rate` to
    minimalny odsetek próbek, na które próg wciąż musi dawać sygnał kierunkowy -
    zabezpiecza przed wybraniem progu tak wysokiego, że model nigdy nic nie
    sygnalizuje (0 sygnałów = niezdefiniowana/myląco "idealna" precyzja).
    """
    if len(records) == 0:
        return 0.5, float("nan")

    conf, pred, true = records[:, 0], records[:, 1], records[:, 2]
    n = len(records)
    best_tau, best_precision = float(thresholds[0]), -1.0

    for tau in thresholds:
        mask = (conf >= tau) & (pred != 0)
        count = mask.sum()
        if count < max(5, min_signal_rate * n):
            continue
        precision = (pred[mask] == true[mask]).mean()
        if precision > best_precision:
            best_precision = precision
            best_tau = float(tau)

    if best_precision < 0:
        return 0.5, float("nan")
    return best_tau, float(best_precision)


@dataclass
class TrainResult:
    model: StackingClassifier
    report: str
    feature_importances: pd.Series
    cv_accuracy: float
    recommended_confidence: float
    recommended_confidence_precision: float
    balanced_accuracy: float
    mcc: float
    auc_roc: float
    expectancy: float
    skipped_folds: int


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 0,
    embargo: int = 0,
    y_reg: pd.Series | None = None,
) -> TrainResult:
    """Trenuje stacking ensemble RF + HistGradientBoosting z walidacją krzyżową
    szeregu czasowego (purged + embargo, bez przecieku danych z przyszłości).

    `gap` pomija próbki między train a test foldem odpowiadające horyzontowi
    etykiety; `embargo` dokłada dodatkowy bufor na końcu train foldu, żeby
    długoterminowe wskaźniki (np. SMA50) też nie "widziały" fragmentu test foldu.

    Foldy, w których train ma mniej niż 2 klasy (nie da się wytrenować
    sensownego klasyfikatora) są pomijane - licznik w `skipped_folds`.

    Dodatkowo, na podstawie zbiorczych (pooled) out-of-fold predykcji ze
    wszystkich foldów:
    - dobiera próg pewności maksymalizujący precyzję sygnałów kierunkowych
      (patrz `_tune_confidence_threshold`) - `recommended_confidence`;
    - liczy `balanced_accuracy`/`mcc`/`auc_roc` (bardziej odporne na przewagę
      liczebną klasy NEUTRALNY niż zwykła trafność) oraz - jeśli podano
      `y_reg` (ciągła stopa zwrotu) - `expectancy`: średni zwrot na transakcję
      dla sygnałów kierunkowych powyżej `recommended_confidence` (dodatni =
      strategia w przeszłości zarabiała więcej niż traciła na takich sygnałach).
    """
    n_splits = min(n_splits, choose_n_splits(len(X)))

    accuracies = []
    last_pred, last_true = None, None
    oof_records = []
    oof_proba = []
    oof_returns = []
    skipped_folds = 0
    for train_idx, test_idx in purged_splits(len(X), n_splits, gap=gap, embargo=embargo):
        if len(train_idx) < MIN_TRAIN_FOLD // 2 or len(test_idx) == 0:
            continue
        if y.iloc[train_idx].nunique() < 2:
            skipped_folds += 1
            continue

        clf = make_classifier()
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = clf.predict(X.iloc[test_idx])
        accuracies.append((pred == y.iloc[test_idx].values).mean())
        last_pred, last_true = pred, y.iloc[test_idx]

        proba = clf.predict_proba(X.iloc[test_idx])
        classes = clf.classes_
        idx = np.argmax(proba, axis=1)
        conf = proba[np.arange(len(idx)), idx]
        fold_pred = classes[idx]
        true_vals = y.iloc[test_idx].values
        oof_records.append(np.column_stack([conf, fold_pred, true_vals]))
        oof_proba.append(_reindex_proba(proba, classes))
        if y_reg is not None:
            oof_returns.append(y_reg.iloc[test_idx].values)

    report = classification_report(
        last_true, last_pred, labels=[-1, 0, 1],
        target_names=["SHORT", "NEUTRALNY", "LONG"], zero_division=0,
    ) if last_true is not None else "Za mało danych na sensowną walidację krzyżową."

    records = np.concatenate(oof_records) if oof_records else np.empty((0, 3))
    recommended_confidence, recommended_precision = _tune_confidence_threshold(records)

    balanced_acc = mcc = auc_roc = expectancy = float("nan")
    if len(records) > 0:
        pred_all, true_all = records[:, 1], records[:, 2]
        balanced_acc = balanced_accuracy_score(true_all, pred_all)
        mcc = matthews_corrcoef(true_all, pred_all)
        try:
            proba_all = np.concatenate(oof_proba)
            auc_roc = roc_auc_score(true_all, proba_all, multi_class="ovr", labels=CANONICAL_CLASSES)
        except ValueError:
            auc_roc = float("nan")  # np. brakuje jednej z klas w zbiorczych danych walidacyjnych

        if oof_returns and not np.isnan(recommended_precision):
            returns_all = np.concatenate(oof_returns)
            conf_all = records[:, 0]
            mask = (conf_all >= recommended_confidence) & (pred_all != 0)
            if mask.sum() > 0:
                payoff = np.where(pred_all[mask] == 1, returns_all[mask], -returns_all[mask])
                expectancy = float(payoff.mean())

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
        recommended_confidence=recommended_confidence,
        recommended_confidence_precision=recommended_precision,
        balanced_accuracy=float(balanced_acc),
        mcc=float(mcc),
        auc_roc=float(auc_roc),
        expectancy=float(expectancy),
        skipped_folds=skipped_folds,
    )


def predict_latest(result: TrainResult, features: pd.DataFrame, feature_cols: list, min_confidence: float | None = None):
    """Zwraca (etykieta_tekstowa, prawdopodobienstwa_dict) dla najnowszej dostępnej świecy.

    `min_confidence`: jeśli podany (np. `result.recommended_confidence`) i
    prawdopodobieństwo najlepszej klasy jest poniżej progu, sygnał jest
    "wyciszany" do NEUTRALNY - model świadomie milczy zamiast dawać słaby,
    nisko-pewny sygnał kierunkowy. `proba` w zwrotce zawsze pokazuje surowe
    prawdopodobieństwa modelu, niezależnie od wyciszenia.
    """
    latest = features.dropna(subset=feature_cols).iloc[[-1]]
    X_latest = latest[feature_cols].astype(float)

    proba = result.model.predict_proba(X_latest)[0]
    classes = result.model.classes_
    idx = int(np.argmax(proba))
    pred = classes[idx]
    confidence = proba[idx]
    proba_dict = {LABELS[c]: float(p) for c, p in zip(classes, proba)}

    if min_confidence is not None and pred != 0 and confidence < min_confidence:
        pred = 0

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
