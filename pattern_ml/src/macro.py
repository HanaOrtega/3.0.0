"""Cechy reżimu całego rynku (nie tylko pojedynczej spółki), wzorowane na
`data/macro.py::derive_market_regime_features` z projektu JuggleLab.

Model oceniający np. AAPL wyłącznie na podstawie AAPL nie "wie", czy cały
rynek jest w hossie, korekcie czy panice - a to silnie wpływa na to, jak
wiarygodny jest sygnał pojedynczej spółki (np. formacja bycza w środku
ogólnorynkowej wyprzedaży jest mniej wiarygodna). Te cechy liczone są z
osobnego indeksu referencyjnego (domyślnie S&P 500, `^GSPC`) i dołączane do
macierzy cech głównego tickera.
"""

import pandas as pd

DEFAULT_BENCHMARK = "^GSPC"


def derive_market_regime_features(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Liczy cechy reżimu rynku z OHLCV indeksu referencyjnego (np. S&P 500)."""
    close = benchmark_df["Close"]
    out = pd.DataFrame(index=benchmark_df.index)
    out["mkt_return_5d"] = close.pct_change(5)
    out["mkt_return_21d"] = close.pct_change(21)
    out["mkt_volatility_21d"] = close.pct_change().rolling(21).std()
    out["mkt_drawdown"] = close / close.cummax() - 1
    out["mkt_trend_63d"] = close.pct_change(63)
    return out


def merge_market_regime(features: pd.DataFrame, market_regime: pd.DataFrame) -> pd.DataFrame:
    """Dołącza cechy reżimu rynku do macierzy cech głównego tickera.

    Indeks referencyjny (benchmark) i główny ticker mogą mieć lekko różne
    kalendarze sesji (różne giełdy/święta) - stąd reindeksacja do indeksu
    `features` z forward-fillem, zamiast prostego złączenia po dokładnej dacie,
    które zgubiłoby wiersze bez idealnie pasującej daty.
    """
    aligned = market_regime.reindex(features.index, method="ffill")
    return features.join(aligned)
