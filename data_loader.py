# src/project/data_loader.py
# Pobieranie danych rynkowych (yfinance) z cache CSV i ujednoliceniem kolumn.

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

# yfinance jest zależnością opcjonalną na etapie importu testów;
# importujemy leniwie wewnątrz funkcji, ale oferujemy jeden prywatny hook.
def _yf_download(symbol: str, start: Optional[str], end: Optional[str], interval: str) -> pd.DataFrame:
    """Thin wrapper na yfinance.download – osobno dla łatwego monkeypatchingu w testach."""
    import yfinance as yf  # lazy import
    return yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )


@dataclass(frozen=True)
class DataLoaderConfig:
    """Ustawienia ładowania danych i cache."""
    interval: str = "1d"
    auto_cache: bool = True
    retries: int = 2
    tz: str = "UTC"  # docelowa strefa czasu indeksu


# ————————————————————————————————————————————————————————————————————————
# Pomocnicze funkcje I/O i walidacji
# ————————————————————————————————————————————————————————————————————————

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "AdjClose": "adj_close",
        "Volume": "volume",
    }
    cols = {c: mapping.get(c, c.lower().replace(" ", "_")) for c in df.columns}
    out = df.rename(columns=cols).copy()
    # tylko interesujące kolumny, jeśli obecne
    wanted = [c for c in ("open", "high", "low", "close", "adj_close", "volume") if c in out.columns]
    out = out[wanted]
    return out


def _ensure_datetime_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        # yfinance zwykle zwraca DatetimeIndex, ale na wszelki wypadek:
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(tz)
    df.index.name = "date"
    return df


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        LOGGER.warning("Nie udało się wczytać cache %s: %s (ignoruję)", path, e)
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _daterange_from_df(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df is None or df.empty:
        return None, None
    start = df.index.min().date().isoformat()
    end = (df.index.max().date()).isoformat()
    return start, end


def _concat_and_dedup(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.sort_index()
    cat = pd.concat([old, new], axis=0)
    cat = cat[~cat.index.duplicated(keep="last")]
    return cat.sort_index()


# ————————————————————————————————————————————————————————————————————————
# Główne API
# ————————————————————————————————————————————————————————————————————————

def _download_with_retries(symbol: str, start: Optional[str], end: Optional[str], interval: str, retries: int) -> pd.DataFrame:
    last_err: Exception | None = None
    for i in range(max(1, retries)):
        try:
            df = _yf_download(symbol, start, end, interval)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
            # Czasami yfinance zwraca pusty DataFrame mimo poprawnego requestu – spróbuj raz jeszcze.
            last_err = RuntimeError("Pusty DataFrame z yfinance")
        except Exception as e:
            last_err = e
        LOGGER.warning("Próba %d pobrania %s nieudana: %s", i + 1, symbol, last_err)
    raise RuntimeError(f"Nie udało się pobrać danych dla {symbol}: {last_err}")


def _prepare_downloaded_df(raw: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = raw.copy()
    # Czasem dla wielu tickerów yfinance zwraca MultiIndex kolumn – spłaszcz
    if isinstance(df.columns, pd.MultiIndex):
        # wybierz pojedynczy ticker jeśli jest
        try:
            # Kolumny w formie (field, ticker)
            df = df.xs(level=-1, axis=1, drop_level=True)
        except Exception:
            # spłaszcz joinem
            df.columns = ["_".join([str(x) for x in tup if x]) for tup in df.columns]
    df = _normalize_columns(df)
    df = _ensure_datetime_index(df, tz)
    # Usuń wiersze, gdzie wszystkie kolumny są NaN
    df = df.dropna(how="all")
    return df


def pobierz_dane_akcji(
    symbol: str,
    data_start: str | None = None,
    data_end: str | None = None,
    interval: str | None = None,
    csv_save_path: str | Path | None = None,
    config: DataLoaderConfig | None = None,
) -> pd.DataFrame:
    """
    Pobiera dane notowań dla `symbol`, aktualizuje cache (jeśli włączone) i zwraca DataFrame.

    Args:
        symbol: np. "AAPL", "GOOGL"
        data_start: ISO data początkowa (np. "2000-01-01") – opcjonalnie
        data_end: ISO data końcowa (np. "2025-08-29") – opcjonalnie
        interval: np. "1d", "1h" – domyślnie z config.interval
        csv_save_path: ścieżka do pliku cache CSV
        config: ustawienia ładowania i cache

    Returns:
        DataFrame z kolumnami: open, high, low, close, adj_close, volume (jeśli dostępne),
        indeks czasu w strefie config.tz, nazwa indeksu 'date'.

    Raises:
        RuntimeError: gdy pobranie danych nie powiedzie się po retry.
        ValueError: gdy symbol pusty.
    """
    if not symbol or not symbol.strip():
        raise ValueError("Symbol nie może być pusty.")
    cfg = config or DataLoaderConfig()
    itv = interval or cfg.interval

    LOGGER.info("--- Calling pobierz_dane_akcji for '%s' ---", symbol)

    # Wczytaj istniejący cache (jeśli wskazano ścieżkę)
    cache_path: Optional[Path] = Path(csv_save_path) if csv_save_path else None
    cached_df = _read_cache(cache_path) if cache_path else None

    # Ustal zakres pobierania:
    # - jeżeli jest cache i auto_cache włączony: pobieramy tylko brakujący ogon
    # - inaczej: pobierz pełny zakres (data_start/data_end)
    dl_start = data_start
    dl_end = data_end
    if cfg.auto_cache and cached_df is not None and not cached_df.empty:
        _, cached_end = _daterange_from_df(cached_df)
        # pobierz dane od (cached_end + 1 dzień) do data_end (lub dziś)
        if cached_end:
            try:
                next_day = (pd.to_datetime(cached_end) + pd.Timedelta(days=1)).date().isoformat()
                dl_start = max(dl_start or next_day, next_day)
            except Exception:
                dl_start = dl_start or cached_end
        # jeśli nie podano data_end – pobieramy do dziś
        dl_end = data_end

    raw = _download_with_retries(symbol, start=dl_start, end=dl_end, interval=itv, retries=cfg.retries)
    new_df = _prepare_downloaded_df(raw, tz=cfg.tz)

    # Jeśli mieliśmy cache – scalamy i deduplikujemy
    final_df = _concat_and_dedup(cached_df, new_df) if cached_df is not None else new_df

    # Zapisz cache, jeśli poproszono
    if cache_path:
        _write_cache(cache_path, final_df)
        LOGGER.info("Successfully %s history to: %s", "saved" if cached_df is None else "updated", cache_path.as_posix())

    LOGGER.info("Successfully downloaded raw data for '%s', shape: %s", symbol, final_df.shape)
    return final_df


def pobierz_dane_sp500(
    data_start: str | None = None,
    data_end: str | None = None,
    interval: str | None = None,
    csv_save_path: str | Path | None = None,
    config: DataLoaderConfig | None = None,
) -> pd.DataFrame:
    """
    Pobiera dane dla indeksu S&P 500 (`^GSPC`) z taką samą polityką cache jak `pobierz_dane_akcji`.
    """
    return pobierz_dane_akcji(
        symbol="^GSPC",
        data_start=data_start,
        data_end=data_end,
        interval=interval,
        csv_save_path=csv_save_path,
        config=config,
    )
