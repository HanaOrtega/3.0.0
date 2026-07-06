"""Klasyczne, reguły-oparte strategie bazowe do porównania z sygnałem ML w backteście.

Logika wzorowana na trzech najprostszych (nie-ML, nie-RL) agentach z
huseinzol05/Stock-Prediction-Models (agent/1.turtle-agent.ipynb,
2.moving-average-agent.ipynb, 3.signal-rolling-agent.ipynb) - z jedną poprawką:
tam okna (rolling window, SMA) liczone są jako procent długości CAŁEGO zbioru
danych (np. "10% z len(df)"), co nie ma sensu przy różnych okresach/interwałach
i w walk-forward. Tutaj okna są stałymi, powszechnie przyjętymi parametrami
(klasyczny 20-dniowy Donchian breakout, SMA 10/30), konfigurowalnymi z CLI.

Strategie są long-only (wejście/wyjście, bez krótkiej sprzedaży) - zgodnie
z oryginałem - w przeciwieństwie do naszej strategii ML, która handluje
też na krótko.
"""

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


def _rolling(values, window: int, fn: str, shift: int = 0) -> np.ndarray:
    series = getattr(pd.Series(values).rolling(window), fn)()
    if shift:
        series = series.shift(shift)
    return series.values


class TurtleStrategy(Strategy):
    """Donchian breakout: kupuj przy wybiciu ponad N-dniowe maksimum, zamykaj
    pozycję przy zejściu poniżej N-dniowego minimum (bez sprzedaży na krótko)."""

    window = 20

    def init(self):
        # shift=1: porównujemy do maks./min. z POPRZEDNICH `window` świec (bez bieżącej) -
        # inaczej rolling max/min zawiera dzisiejszą świecę i wybicie nigdy by się nie zaliczyło
        self.upper = self.I(_rolling, self.data.High, self.window, "max", 1)
        self.lower = self.I(_rolling, self.data.Low, self.window, "min", 1)

    def next(self):
        if np.isnan(self.upper[-1]) or np.isnan(self.lower[-1]):
            return
        price = self.data.Close[-1]
        if not self.position and price > self.upper[-1]:
            self.buy()
        elif self.position and price < self.lower[-1]:
            self.position.close()


class SmaCrossStrategy(Strategy):
    """Klasyczny crossover szybkiej/wolnej średniej kroczącej."""

    fast = 10
    slow = 30

    def init(self):
        self.sma_fast = self.I(_rolling, self.data.Close, self.fast, "mean")
        self.sma_slow = self.I(_rolling, self.data.Close, self.slow, "mean")

    def next(self):
        if np.isnan(self.sma_fast[-1]) or np.isnan(self.sma_slow[-1]):
            return
        if crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast) and self.position:
            self.position.close()


class ContrarianStrategy(Strategy):
    """Kupuj po `delay` kolejnych spadkach, sprzedawaj po `delay` kolejnych
    wzrostach - strategia "kup dołek, sprzedaj górkę" z potwierdzeniem
    opóźnieniem (unika whipsawów)."""

    delay = 3

    def init(self):
        self._counter = 0

    def next(self):
        if len(self.data.Close) < 2:
            return
        price, prev = self.data.Close[-1], self.data.Close[-2]

        if price < prev and not self.position:
            self._counter += 1
            if self._counter >= self.delay:
                self.buy()
                self._counter = 0
        elif price > prev and self.position:
            self._counter += 1
            if self._counter >= self.delay:
                self.position.close()
                self._counter = 0
        else:
            self._counter = 0


BASELINES = {
    "turtle": TurtleStrategy,
    "sma": SmaCrossStrategy,
    "contrarian": ContrarianStrategy,
}


def run_baseline_backtest(
    df: pd.DataFrame,
    name: str,
    cash: float = 10_000,
    commission: float = 0.0007,
    **params,
):
    """Uruchamia jedną z prostych strategii bazowych (`turtle`, `sma`, `contrarian`)."""
    if name not in BASELINES:
        raise ValueError(f"Nieznana strategia bazowa '{name}'. Dostępne: {list(BASELINES)}")

    base_cls = BASELINES[name]
    strategy_cls = type(f"Configured{base_cls.__name__}", (base_cls,), params)
    bt = Backtest(
        df, strategy_cls, cash=cash, commission=commission,
        exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run()
    return bt, stats
