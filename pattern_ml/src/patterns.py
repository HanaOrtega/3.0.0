"""Rozpoznawanie klasycznych formacji świecowych (reguły oparte o kształt świec).

Każda funkcja zwraca maskę bool (pd.Series) - True tam, gdzie formacja występuje
na ostatniej świecy danego okna. Funkcja `detect_all` łączy wszystkie formacje
w jedną tabelę oraz tworzy skumulowany sygnał kierunkowy (+1 bycza / -1 niedźwiedzia).
"""

import numpy as np
import pandas as pd

# Formacje byków / niedźwiedzi - używane do zbudowania sygnału kierunkowego
BULLISH = [
    "hammer",
    "bullish_engulfing",
    "morning_star",
    "piercing_line",
    "three_white_soldiers",
    "inverted_hammer",
]
BEARISH = [
    "shooting_star",
    "bearish_engulfing",
    "evening_star",
    "dark_cloud_cover",
    "three_black_crows",
    "hanging_man",
]


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["Close"] - df["Open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return (df["High"] - df["Low"]).replace(0, np.nan)


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df[["Open", "Close"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["Open", "Close"]].min(axis=1) - df["Low"]


def _is_bullish_candle(df: pd.DataFrame) -> pd.Series:
    return df["Close"] > df["Open"]


def _is_bearish_candle(df: pd.DataFrame) -> pd.Series:
    return df["Close"] < df["Open"]


def _trend_context(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Prosty kontekst trendu: nachylenie SMA przed świecą (>0 = trend wzrostowy)."""
    sma = df["Close"].rolling(window).mean()
    return sma.diff()


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca DataFrame z kolumnami bool dla każdej formacji + kolumnę 'pattern_signal'."""
    body = _body(df)
    rng = _range(df)
    upper = _upper_shadow(df)
    lower = _lower_shadow(df)
    bull_candle = _is_bullish_candle(df)
    bear_candle = _is_bearish_candle(df)
    trend = _trend_context(df)

    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    prev_body = body.shift(1)
    prev_bull = bull_candle.shift(1)
    prev_bear = bear_candle.shift(1)

    out = pd.DataFrame(index=df.index)

    # --- Formacje jednoświecowe ---
    small_body = body <= 0.1 * rng
    out["doji"] = small_body.fillna(False)

    out["hammer"] = (
        (lower >= 2 * body)
        & (upper <= 0.3 * body.replace(0, np.nan))
        & (body > 0)
        & (trend < 0)  # po trendzie spadkowym
    ).fillna(False)

    out["inverted_hammer"] = (
        (upper >= 2 * body)
        & (lower <= 0.3 * body.replace(0, np.nan))
        & (body > 0)
        & (trend < 0)
    ).fillna(False)

    out["hanging_man"] = (
        (lower >= 2 * body)
        & (upper <= 0.3 * body.replace(0, np.nan))
        & (body > 0)
        & (trend > 0)  # po trendzie wzrostowym
    ).fillna(False)

    out["shooting_star"] = (
        (upper >= 2 * body)
        & (lower <= 0.3 * body.replace(0, np.nan))
        & (body > 0)
        & (trend > 0)
    ).fillna(False)

    # --- Formacje dwuświecowe ---
    out["bullish_engulfing"] = (
        prev_bear
        & bull_candle
        & (df["Open"] <= prev_close)
        & (df["Close"] >= prev_open)
        & (body > prev_body)
    ).fillna(False)

    out["bearish_engulfing"] = (
        prev_bull
        & bear_candle
        & (df["Open"] >= prev_close)
        & (df["Close"] <= prev_open)
        & (body > prev_body)
    ).fillna(False)

    prev_mid = (prev_open + prev_close) / 2
    out["piercing_line"] = (
        prev_bear
        & bull_candle
        & (df["Open"] < prev_close)
        & (df["Close"] > prev_mid)
        & (df["Close"] < prev_open)
    ).fillna(False)

    out["dark_cloud_cover"] = (
        prev_bull
        & bear_candle
        & (df["Open"] > prev_close)
        & (df["Close"] < prev_mid)
        & (df["Close"] > prev_open)
    ).fillna(False)

    # --- Formacje trzyświecowe ---
    o1, c1 = df["Open"].shift(2), df["Close"].shift(2)
    o2, c2 = df["Open"].shift(1), df["Close"].shift(1)
    o3, c3 = df["Open"], df["Close"]

    first_bear = c1 < o1
    first_bull = c1 > o1
    small_middle = (c2 - o2).abs() <= 0.3 * (c1 - o1).abs().replace(0, np.nan)
    third_bull = c3 > o3
    third_bear = c3 < o3

    out["morning_star"] = (
        first_bear
        & small_middle
        & third_bull
        & (c3 > (o1 + c1) / 2)
    ).fillna(False)

    out["evening_star"] = (
        first_bull
        & small_middle
        & third_bear
        & (c3 < (o1 + c1) / 2)
    ).fillna(False)

    out["three_white_soldiers"] = (
        (c1 > o1) & (c2 > o2) & (c3 > o3)
        & (c2 > c1) & (c3 > c2)
        & (o2 > o1) & (o2 < c1)
        & (o3 > o2) & (o3 < c2)
    ).fillna(False)

    out["three_black_crows"] = (
        (c1 < o1) & (c2 < o2) & (c3 < o3)
        & (c2 < c1) & (c3 < c2)
        & (o2 < o1) & (o2 > c1)
        & (o3 < o2) & (o3 > c2)
    ).fillna(False)

    # --- Skumulowany sygnał kierunkowy formacji (-1, 0, +1) ---
    bullish_hit = out[BULLISH].any(axis=1)
    bearish_hit = out[BEARISH].any(axis=1)
    out["pattern_signal"] = np.select(
        [bullish_hit & ~bearish_hit, bearish_hit & ~bullish_hit],
        [1, -1],
        default=0,
    )

    return out
