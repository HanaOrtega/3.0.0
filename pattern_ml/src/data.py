"""Pobieranie danych OHLCV z yfinance."""

import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Pobiera dane świecowe (Open/High/Low/Close/Volume) dla danego tickera."""
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"Brak danych dla tickera '{ticker}' (period={period}, interval={interval})."
        )

    # yfinance czasem zwraca kolumny MultiIndex (np. przy jednym tickerze w liście)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    df.index.name = "Date"
    keep = ["Open", "High", "Low", "Close", "Volume"]
    return df[keep].dropna()
