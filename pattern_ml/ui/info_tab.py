"""Zakładka GUI: informacje o platformie i zastrzeżenia."""

import streamlit as st


def render():
    st.subheader("O platformie")
    st.markdown(
        """
Ta platforma spina w jednym miejscu wszystkie moduły projektu:

- **Sygnał ML** — pobranie danych z yfinance, wykrywanie formacji świecowych,
  wskaźniki analizy technicznej, stacking ensemble ML (RandomForest +
  HistGradientBoosting) przewidujący kierunek (LONG/SHORT/NEUTRALNY) oraz
  regresory kwantylowe rysujące prognozowany stożek ceny na wykresie.
- **Backtest** — pełny walk-forward backtest tego sygnału z realistyczną
  egzekucją (prowizje, ATR stop-loss/take-profit, kill-switch), z opcjonalnym
  porównaniem do klasycznych strategii bazowych (Donchian breakout, SMA
  crossover, kontrariańska).
- **Sentyment z X** — świeże (nie archiwalne) wzmianki o instrumencie z X
  (Twitter), agregowane do dziennych cech i opcjonalnie dołączane do modelu ML.

Szczegółowy opis każdego modułu, parametrów CLI i decyzji projektowych
znajduje się w `README.md` w katalogu `pattern_ml/`.
        """
    )

    st.warning(
        "To narzędzie edukacyjne/analityczne, nie system automatycznego handlu. "
        "Skuteczność modelu zawsze warto sprawdzić w zakładce Backtest przed "
        "podjęciem jakiejkolwiek decyzji inwestycyjnej - rynki finansowe są w dużej "
        "mierze losowe i żaden model nie daje gwarancji."
    )
    st.info(
        "Automatyczne logowanie/scrapowanie X podlega regulaminowi platformy (X Terms "
        "of Service) - używaj tej funkcji na własnym koncie, w rozsądnych odstępach "
        "czasu i wyłącznie do własnych celów analitycznych."
    )
