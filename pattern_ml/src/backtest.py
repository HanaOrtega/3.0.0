"""Pełny walk-forward backtest sygnału ML (formacje świecowe + analiza techniczna).

Silnik egzekucji zleceń i metryki (Sharpe, Sortino, max drawdown, win rate,
profit factor, equity curve...) pochodzą z biblioteki `backtesting.py`
(kernc/backtesting.py) - dojrzałego, szeroko używanego narzędzia, zamiast
własnej, podatnej na błędy implementacji tych wzorów.

Kluczowa zasada (wzorowana na oficjalnym przykładzie tej biblioteki "Trading
with Machine Learning" oraz na koncepcji "purgingu" z prac Lopeza de Prado
o walidacji modeli ML w finansach): model jest retrenowany co `retrain_every`
świec na kroczącym oknie (`train_window`) i w każdym momencie widzi WYŁĄCZNIE
dane do bieżącej świecy włącznie - `backtesting.py` samo obcina `self.data`
do historii "widzianej do teraz", a etykiety (`build_feature_matrix`) same
odrzucają ostatnie `horizon` wierszy (bo ich etykieta zależałaby od cen
jeszcze nieznanych) - dzięki temu model nigdy nie trenuje ani nie przewiduje
na podstawie danych z przyszłości względem symulowanego momentu decyzji.
"""

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from .features import build_feature_matrix
from .model import make_classifier

MIN_WARMUP_BARS = 60  # najdłuższe okno wskaźnika (SMA50) + zapas na NaN z ta-lib


class MLStrategy(Strategy):
    # parametry modelu / etykiety - takie same znaczenie jak w features.build_feature_matrix
    horizon = 5
    atr_mult = 0.5

    # parametry walk-forward
    retrain_every = 20
    train_window = 250

    # parametry decyzji i zarządzania ryzykiem
    min_confidence = 0.40
    risk_pct = 0.01
    sl_atr_mult = 1.5
    tp_atr_mult = 2.5

    def init(self):
        self._model = None
        self._last_train_len = -1

    def next(self):
        n = len(self.data.df)
        if n < MIN_WARMUP_BARS + self.horizon:
            return

        if self._model is None or (n - self._last_train_len) >= self.retrain_every:
            self._retrain(n)
        if self._model is None:
            return

        start = max(0, n - self.train_window)
        df_slice = self.data.df.iloc[start:]
        features, _, _, _, _, feature_cols = build_feature_matrix(
            df_slice, horizon=self.horizon, atr_mult=self.atr_mult
        )
        valid = features.dropna(subset=feature_cols)
        if valid.empty:
            return

        X_latest = valid[feature_cols].astype(float).iloc[[-1]]
        atr = valid["atr"].iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            return

        proba = self._model.predict_proba(X_latest)[0]
        classes = self._model.classes_
        idx = int(np.argmax(proba))
        pred_class = classes[idx]
        confidence = proba[idx]

        if confidence < self.min_confidence:
            return

        price = self.data.Close[-1]
        sl_dist = self.sl_atr_mult * atr
        tp_dist = self.tp_atr_mult * atr
        fraction = float(np.clip(self.risk_pct / (sl_dist / price), 0.01, 0.99))

        if pred_class == 1:
            if not self.position.is_long:
                self.buy(size=fraction, sl=price - sl_dist, tp=price + tp_dist)
        elif pred_class == -1:
            if not self.position.is_short:
                self.sell(size=fraction, sl=price + sl_dist, tp=price - tp_dist)
        elif self.position:
            self.position.close()

    def _retrain(self, n: int) -> None:
        start = max(0, n - self.train_window)
        df_slice = self.data.df.iloc[start:]
        _, _, X, y, _, _ = build_feature_matrix(df_slice, horizon=self.horizon, atr_mult=self.atr_mult)
        if len(X) < MIN_WARMUP_BARS or y.nunique() < 2:
            return
        clf = make_classifier()
        clf.fit(X, y)
        self._model = clf
        self._last_train_len = n


def run_backtest(
    df: pd.DataFrame,
    horizon: int = 5,
    atr_mult: float = 0.5,
    retrain_every: int = 20,
    train_window: int = 250,
    min_confidence: float = 0.40,
    risk_pct: float = 0.01,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    cash: float = 10_000,
    commission: float = 0.0007,
):
    """Uruchamia walk-forward backtest i zwraca (obiekt Backtest, statystyki)."""
    strategy_cls = type(
        "ConfiguredMLStrategy",
        (MLStrategy,),
        dict(
            horizon=horizon,
            atr_mult=atr_mult,
            retrain_every=retrain_every,
            train_window=train_window,
            min_confidence=min_confidence,
            risk_pct=risk_pct,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
        ),
    )
    bt = Backtest(df, strategy_cls, cash=cash, commission=commission, exclusive_orders=True)
    stats = bt.run()
    return bt, stats
