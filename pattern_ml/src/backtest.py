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

from .features import MAX_INDICATOR_LOOKBACK, build_feature_matrix
from .model import make_classifier, train_quantile_models

MIN_WARMUP_BARS = MAX_INDICATOR_LOOKBACK + 10  # najdłuższe okno cechy (fractional diff) + zapas na NaN z ta-lib


class MLStrategy(Strategy):
    # parametry modelu / etykiety - takie same znaczenie jak w features.build_feature_matrix
    horizon = 5
    atr_mult = 0.5
    market_regime = None  # opcjonalnie: DataFrame z src.macro.derive_market_regime_features

    # parametry walk-forward
    retrain_every = 20
    train_window = 300

    # parametry decyzji i zarządzania ryzykiem
    min_confidence = 0.40
    risk_pct = 0.01
    sl_atr_mult = 1.5
    tp_atr_mult = 2.5
    # "quantile": SL/TP wyznaczane z rozrzutu prognozy P10/P90 (szersza niepewność
    # modelu = szerszy stop) z podłogą ATR na wypadek zdegenerowanego/wąskiego
    # stożka; "atr": zawsze stałe wielokrotności ATR (stare zachowanie)
    sizing_mode = "quantile"

    # kill-switch: wstrzymuje otwieranie NOWYCH pozycji (nie zamyka istniejących -
    # o to dba SL/TP), gdy strategia wpadnie w wyraźną serię strat, chroniąc przed
    # dalszym "uporczywym" handlem w reżimie, w którym model wyraźnie się myli
    max_drawdown_halt = 0.25
    loss_streak_halt = 5

    def init(self):
        self._model = None
        self._quantile_models = None
        self._last_train_len = -1
        self._peak_equity = self.equity

    def _kill_switch_active(self) -> bool:
        self._peak_equity = max(self._peak_equity, self.equity)
        drawdown = 1 - self.equity / self._peak_equity
        if drawdown > self.max_drawdown_halt:
            return True

        recent = self.closed_trades[-self.loss_streak_halt:]
        if len(recent) >= self.loss_streak_halt and all(t.pl < 0 for t in recent):
            return True

        return False

    def _market_regime_slice(self, df_slice: pd.DataFrame):
        if self.market_regime is None:
            return None
        return self.market_regime.reindex(df_slice.index, method="ffill")

    def _risk_distances(self, pred_class: int, price: float, atr: float, X_latest: pd.DataFrame):
        """Zwraca (sl_dist, tp_dist). W trybie 'quantile' korzysta z rozrzutu
        prognozy P10/P90 regresorów kwantylowych (szersza niepewność modelu co
        do przyszłej ceny -> szerszy stop), z podłogą 0.5x ATR, żeby uniknąć
        zbyt ciasnych stopów przy zdegenerowanym/wąskim stożku prognozy."""
        atr_sl, atr_tp = self.sl_atr_mult * atr, self.tp_atr_mult * atr
        if self.sizing_mode != "quantile" or self._quantile_models is None:
            return atr_sl, atr_tp

        q_returns = {q: model.predict(X_latest)[0] for q, model in self._quantile_models.items()}
        floor = 0.5 * atr

        if pred_class == 1:
            sl_dist = max(price - price * (1 + q_returns[0.1]), floor)
            tp_dist = max(price * (1 + q_returns[0.9]) - price, floor)
        else:
            sl_dist = max(price * (1 + q_returns[0.9]) - price, floor)
            tp_dist = max(price - price * (1 + q_returns[0.1]), floor)

        return sl_dist, tp_dist

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
            df_slice, horizon=self.horizon, atr_mult=self.atr_mult,
            market_regime=self._market_regime_slice(df_slice),
            preselected_features=self._feature_cols, quiet=True,
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

        if pred_class == 0 and self.position:
            self.position.close()
            return
        if pred_class == 0 or self._kill_switch_active():
            return

        price = self.data.Close[-1]
        sl_dist, tp_dist = self._risk_distances(pred_class, price, atr, X_latest)
        fraction = float(np.clip(self.risk_pct / (sl_dist / price), 0.01, 0.99))

        if pred_class == 1 and not self.position.is_long:
            self.buy(size=fraction, sl=price - sl_dist, tp=price + tp_dist)
        elif pred_class == -1 and not self.position.is_short:
            self.sell(size=fraction, sl=price + sl_dist, tp=price - tp_dist)

    def _retrain(self, n: int) -> None:
        start = max(0, n - self.train_window)
        df_slice = self.data.df.iloc[start:]
        _, _, X, y, y_reg, feature_cols = build_feature_matrix(
            df_slice, horizon=self.horizon, atr_mult=self.atr_mult,
            market_regime=self._market_regime_slice(df_slice), quiet=True,
        )
        if len(X) < MIN_WARMUP_BARS or y.nunique() < 2:
            return
        clf = make_classifier()
        clf.fit(X, y)
        self._model = clf
        self._feature_cols = list(feature_cols)
        self._quantile_models = train_quantile_models(X, y_reg) if self.sizing_mode == "quantile" else None
        self._last_train_len = n


def run_backtest(
    df: pd.DataFrame,
    horizon: int = 5,
    atr_mult: float = 0.5,
    retrain_every: int = 20,
    train_window: int = 300,
    min_confidence: float = 0.40,
    risk_pct: float = 0.01,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    sizing_mode: str = "quantile",
    max_drawdown_halt: float = 0.25,
    loss_streak_halt: int = 5,
    market_regime: pd.DataFrame | None = None,
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
            sizing_mode=sizing_mode,
            max_drawdown_halt=max_drawdown_halt,
            loss_streak_halt=loss_streak_halt,
            market_regime=market_regime,
        ),
    )
    bt = Backtest(
        df, strategy_cls, cash=cash, commission=commission,
        exclusive_orders=True, finalize_trades=True,
    )
    stats = bt.run()
    return bt, stats
