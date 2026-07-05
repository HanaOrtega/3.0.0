"""Budowa cech (features) do modelu ML: wskaźniki analizy technicznej + formacje świecowe."""

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from .patterns import detect_all


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


def build_feature_matrix(df: pd.DataFrame, horizon: int = 5, atr_mult: float = 0.5):
    """Łączy wskaźniki TA + formacje świecowe w macierz cech X oraz etykiety y.

    Etykieta klasyfikacyjna (y): kierunek ceny za `horizon` świec, w 3 klasach:
      1  = LONG  (wzrost > atr_mult * ATR)
     -1  = SHORT (spadek > atr_mult * ATR)
      0  = NEUTRALNY (ruch w granicach szumu rynkowego)

    Etykieta regresyjna (y_reg): ciągła stopa zwrotu za `horizon` świec - używana
    do wyznaczenia prognozowanej ścieżki ceny (stożka niepewności) na wykresie.

    Zwraca: (features_df, patterns_df, X, y, y_reg, feature_cols).
    """
    ind = build_indicators(df)
    pat = detect_all(df)
    shape = _candle_shape_features(df)

    features = pd.concat([ind, pat.drop(columns=["pattern_signal"]), shape], axis=1)
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
        "pattern_signal",
    ] + list(pat.drop(columns=["pattern_signal"]).columns)

    model_data = features.dropna(subset=feature_cols)
    train_data = model_data.dropna(subset=["label", "future_return"])

    X = train_data[feature_cols].astype(float)
    y = train_data["label"].astype(int)
    y_reg = train_data["future_return"].astype(float)

    return features, pat, X, y, y_reg, feature_cols
