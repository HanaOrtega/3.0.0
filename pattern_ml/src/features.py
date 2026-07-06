"""Budowa cech (features) do modelu ML: wskaźniki analizy technicznej + formacje świecowe."""

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from .feature_selection import select_features
from .macro import merge_market_regime
from .patterns import detect_all
from .sentiment import merge_sentiment_features

AGGRESSIVE_CLEANING_ROW_THRESHOLD = 200
AGGRESSIVE_CLEANING_DROP_FRACTION = 0.25
MAX_INDICATOR_LOOKBACK = 100  # okno fractional differencing (_fractional_difference) - najdłuższe spośród cech


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Dokłada do df kolumny z klasycznymi wskaźnikami analizy technicznej."""
    out = df.copy()

    out["sma20"] = SMAIndicator(out["Close"], window=20).sma_indicator()
    out["sma50"] = SMAIndicator(out["Close"], window=50).sma_indicator()
    out["ema12"] = EMAIndicator(out["Close"], window=12).ema_indicator()
    out["ema26"] = EMAIndicator(out["Close"], window=26).ema_indicator()

    macd = MACD(out["Close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    out["rsi"] = RSIIndicator(out["Close"], window=14).rsi()

    stoch = StochasticOscillator(out["High"], out["Low"], out["Close"])
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    out["williams_r"] = WilliamsRIndicator(out["High"], out["Low"], out["Close"]).williams_r()
    out["cci"] = CCIIndicator(out["High"], out["Low"], out["Close"], window=20).cci()
    out["roc"] = ROCIndicator(out["Close"], window=10).roc()

    bb = BollingerBands(out["Close"], window=20)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_pct"] = bb.bollinger_pband()
    out["bb_width"] = bb.bollinger_wband()

    atr = AverageTrueRange(out["High"], out["Low"], out["Close"], window=14)
    out["atr"] = atr.average_true_range()
    out["atr_pct"] = out["atr"] / out["Close"]

    adx = ADXIndicator(out["High"], out["Low"], out["Close"], window=14)
    out["adx"] = adx.adx()

    obv = OnBalanceVolumeIndicator(out["Close"], out["Volume"]).on_balance_volume()
    out["obv_change"] = obv.pct_change(10)
    out["volume_change"] = out["Volume"].pct_change()
    out["volume_sma20"] = out["Volume"].rolling(20).mean()
    out["volume_vs_sma20"] = out["Volume"] / out["volume_sma20"] - 1

    return out


def _fracdiff_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
    weights = [1.0]
    for k in range(1, size):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    return np.array(weights[::-1])


def _fractional_difference(series: pd.Series, d: float = 0.4, threshold: float = 1e-4, max_window: int = 100) -> pd.Series:
    """Różnicowanie frakcyjne (Lopez de Prado) log-ceny: usuwa trend (stacjonarność
    potrzebna modelowi drzewiastemu, żeby nie "uczyć się" konkretnego poziomu ceny),
    ale w przeciwieństwie do zwykłego `pct_change()` zachowuje część pamięci długoterminowej
    (wagi maleją, ale nie znikają nagle) - stąd bogatszy sygnał niż proste stopy zwrotu."""
    weights = _fracdiff_weights(d, max_window, threshold)
    width = len(weights)
    return series.rolling(width).apply(lambda x: np.dot(weights, x), raw=True)


def _hurst_exponent(x: np.ndarray, max_lag: int = 15) -> float:
    """Szacuje wykładnik Hursta (rozrzut wariancji przyrostów w skali log-log):
    <0.5 = reżim mean-reverting, ~0.5 = błądzenie losowe, >0.5 = reżim trendujący."""
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    tau = np.where(tau > 1e-8, tau, 1e-8)
    slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(slope * 2.0)


def _rolling_entropy(returns: pd.Series, window: int = 30, bins: int = 8) -> pd.Series:
    """Entropia Shannona rozkładu zwrotów w oknie - niska = uporządkowany/kierunkowy
    ruch ceny, wysoka = szum bez wyraźnego kierunku."""
    def entropy_fn(x):
        hist, _ = np.histogram(x, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    return returns.rolling(window).apply(entropy_fn, raw=True)


def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cechy opisujące "reżim" rynku (trend/mean-reversion/szum), wzorowane na
    feature_enrichment.py (fractional differencing) i custom_features.py
    (wykładnik Hursta, entropia, rozkład zwrotów) z projektu JuggleLab."""
    out = pd.DataFrame(index=df.index)
    log_close = np.log(df["Close"])
    returns = df["Close"].pct_change()
    out["frac_diff_close"] = _fractional_difference(log_close)
    out["hurst"] = log_close.rolling(40).apply(_hurst_exponent, raw=True)
    out["return_entropy"] = _rolling_entropy(returns, window=20)
    out["return_skew"] = returns.rolling(20).skew()
    out["return_kurt"] = returns.rolling(20).kurt()
    out["return_mad"] = returns.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return out


def _candle_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cechy ciągłe opisujące kształt świecy (silniejszy sygnał dla ML niż same flagi formacji)."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    body = (df["Close"] - df["Open"]).abs()
    upper = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower = df[["Open", "Close"]].min(axis=1) - df["Low"]

    out = pd.DataFrame(index=df.index)
    out["body_pct"] = body / rng
    out["upper_shadow_pct"] = upper / rng
    out["lower_shadow_pct"] = lower / rng
    out["candle_direction"] = np.sign(df["Close"] - df["Open"])
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int = 5,
    atr_mult: float = 0.5,
    sentiment: pd.DataFrame | None = None,
    market_regime: pd.DataFrame | None = None,
    auto_select_features: bool = True,
    preselected_features: list | None = None,
    quiet: bool = False,
):
    """Łączy wskaźniki TA + formacje świecowe (+ opcjonalnie sentyment z X i reżim
    rynku) w macierz cech X oraz etykiety y.

    Etykieta klasyfikacyjna (y): kierunek ceny za `horizon` świec, w 3 klasach:
      1  = LONG  (wzrost > atr_mult * ATR)
     -1  = SHORT (spadek > atr_mult * ATR)
      0  = NEUTRALNY (ruch w granicach szumu rynkowego)

    Etykieta regresyjna (y_reg): ciągła stopa zwrotu za `horizon` świec - używana
    do wyznaczenia prognozowanej ścieżki ceny (stożka niepewności) na wykresie.

    `sentiment` (opcjonalnie): wynik `src.sentiment.load_x_sentiment()` - dzienne
    cechy świeżych wzmianek z X, dołączane jako dodatkowe kolumny `sent_*`.

    `market_regime` (opcjonalnie): wynik `src.macro.derive_market_regime_features()`
    - cechy reżimu całego rynku (indeksu referencyjnego), kolumny `mkt_*`.

    `auto_select_features`: odrzuca cechy prawie stałe lub nadmiarowe (silnie
    skorelowane z inną) - patrz `src.feature_selection.select_features`. Filtr
    patrzy wyłącznie na same cechy (nie na etykietę), więc jest bezpieczny
    względem przecieku danych. Ignorowane, jeśli podano `preselected_features`.

    `preselected_features` (opcjonalnie): użyj DOKŁADNIE tej listy kolumn
    zamiast auto-selekcji - potrzebne przy predykcji na modelu już
    wytrenowanym na konkretnym zestawie cech (np. w walk-forward backteście,
    gdzie okno danych przesuwa się co świecę, więc świeża auto-selekcja
    mogłaby dać inny zestaw kolumn niż ten, na którym model faktycznie się uczył).

    Zwraca: (features_df, patterns_df, X, y, y_reg, feature_cols) - `feature_cols`
    to lista PO selekcji (jeśli włączona).
    """
    ind = build_indicators(df)
    pat = detect_all(df)
    shape = _candle_shape_features(df)
    regime = _regime_features(df)

    features = pd.concat([ind, pat.drop(columns=["pattern_signal"]), shape, regime], axis=1)
    features["pattern_signal"] = pat["pattern_signal"]

    # cechy relatywne (bardziej stabilne dla ML niż ceny bezwzględne)
    features["close_vs_sma20"] = features["Close"] / features["sma20"] - 1
    features["close_vs_sma50"] = features["Close"] / features["sma50"] - 1
    features["sma20_vs_sma50"] = features["sma20"] / features["sma50"] - 1
    features["ema_cross"] = features["ema12"] / features["ema26"] - 1
    features["return_1"] = features["Close"].pct_change(1)
    features["return_5"] = features["Close"].pct_change(5)
    features["return_10"] = features["Close"].pct_change(10)
    features["volatility_10"] = features["return_1"].rolling(10).std()

    future_return = features["Close"].shift(-horizon) / features["Close"] - 1
    threshold = atr_mult * features["atr_pct"]
    label = np.select(
        [future_return > threshold, future_return < -threshold],
        [1, -1],
        default=0,
    )
    features["label"] = label
    features["future_return"] = future_return
    features.loc[features.index[-horizon:], ["label", "future_return"]] = np.nan

    feature_cols = [
        "rsi", "stoch_k", "stoch_d", "williams_r", "cci", "roc",
        "macd", "macd_signal", "macd_diff",
        "bb_pct", "bb_width", "atr_pct", "adx",
        "obv_change", "volume_change", "volume_vs_sma20",
        "close_vs_sma20", "close_vs_sma50", "sma20_vs_sma50", "ema_cross",
        "return_1", "return_5", "return_10", "volatility_10",
        "body_pct", "upper_shadow_pct", "lower_shadow_pct", "candle_direction",
        "frac_diff_close", "hurst", "return_entropy", "return_skew", "return_kurt", "return_mad",
        "pattern_signal",
    ] + list(pat.drop(columns=["pattern_signal"]).columns)

    if sentiment is not None:
        features = merge_sentiment_features(features, sentiment)
        feature_cols += ["sent_mentions", "sent_engagement", "sent_polarity", "sent_days_since_mention"]

    if market_regime is not None:
        features = merge_market_regime(features, market_regime)
        feature_cols += list(market_regime.columns)

    model_data = features.dropna(subset=feature_cols)
    train_data = model_data.dropna(subset=["label", "future_return"])

    # ostrzeżenie zamiast cichego trenowania na okrojonym zbiorze (wzorem
    # data_cleaning.py z JuggleLab): wskaźniki TA nigdy nie są uzupełniane
    # sztucznie (ffill/bfill) - wiersze z NaN są odrzucane, ale jeśli to
    # oznacza utratę dużej części małego zbioru, użytkownik powinien o tym wiedzieć
    if not quiet and len(features) < AGGRESSIVE_CLEANING_ROW_THRESHOLD and len(features) > 0:
        dropped_fraction = 1 - len(model_data) / len(features)
        if dropped_fraction > AGGRESSIVE_CLEANING_DROP_FRACTION:
            print(
                f"UWAGA: {dropped_fraction * 100:.0f}% wierszy odrzucono z powodu brakujących "
                f"wskaźników (rozgrzewka SMA/EMA) na zbiorze zaledwie {len(features)} świec - "
                "rozważ dłuższy --period, żeby model miał więcej danych treningowych."
            )

    X = train_data[feature_cols].astype(float)
    y = train_data["label"].astype(int)
    y_reg = train_data["future_return"].astype(float)

    if preselected_features is not None:
        feature_cols = [c for c in preselected_features if c in X.columns]
        X = X[feature_cols]
    elif auto_select_features and not X.empty:
        feature_cols = select_features(X)
        X = X[feature_cols]

    return features, pat, X, y, y_reg, feature_cols
