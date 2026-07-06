"""Pobieranie danych OHLCV z yfinance.

Odporność na typowe problemy yfinance (wzorowane na data/fetching.py z projektu
JuggleLab): ponawianie z wykładniczym backoffem przy chwilowych błędach sieci,
przycinanie `period` do realnych limitów yfinance dla danych śróddziennych
(inaczej zapytanie po cichu zwraca puste/obcięte dane), oraz normalizacja
strefy czasowej indeksu (yfinance zwraca czasem tz-aware, czasem tz-naive
w zależności od giełdy/interwału).
"""

import time

import pandas as pd
import yfinance as yf

# Twarde limity yfinance na długość historii dla danych śróddziennych (w dniach)
INTRADAY_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 60, "1h": 730,
}

_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
    "1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "ytd": 365, "max": 10_000,
}


def _clamp_intraday_period(period: str, interval: str) -> str:
    """Przycina `period` do realnego limitu yfinance dla danego interwału
    śróddziennego - bez tego zapytanie poza limitem zwraca ciszej obcięte
    lub puste dane zamiast błędu."""
    max_days = INTRADAY_MAX_DAYS.get(interval)
    if max_days is None:
        return period
    requested_days = _PERIOD_DAYS.get(period, max_days)
    if requested_days <= max_days:
        return period
    clamped = f"{max_days}d"
    print(
        f"UWAGA: interwał '{interval}' wspiera maks. {max_days} dni historii - "
        f"przycinam --period z '{period}' do '{clamped}'."
    )
    return clamped


def _download_with_retry(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    delay = 1.0
    last_exc = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=False, progress=False,
            )
            if not df.empty:
                return df
        except Exception as exc:  # połączenie/timeout/HTTP błąd yfinance
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    if last_exc:
        raise ConnectionError(f"Pobieranie danych dla '{ticker}' nie powiodło się: {last_exc}") from last_exc
    return pd.DataFrame()


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Pobiera dane świecowe (Open/High/Low/Close/Volume) dla danego tickera."""
    period = _clamp_intraday_period(period, interval)
    df = _download_with_retry(ticker, period, interval)

    if df.empty:
        raise ValueError(
            f"Brak danych dla tickera '{ticker}' (period={period}, interval={interval})."
        )

    # yfinance czasem zwraca kolumny MultiIndex (np. przy jednym tickerze w liście)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    keep = ["Open", "High", "Low", "Close", "Volume"]
    return df[keep].dropna()
