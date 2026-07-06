"""Zakładka GUI: pełny walk-forward backtest sygnału ML + porównanie z baseline'ami."""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.backtest import run_backtest
from src.baselines import BASELINES, run_baseline_backtest
from src.macro import DEFAULT_BENCHMARK, derive_market_regime_features
from src.quality import DataQualityError, validate_ohlcv

from .common import cached_fetch_ohlcv

SUMMARY_COLS = [
    "Return [%]", "Buy & Hold Return [%]", "Sharpe Ratio", "Max. Drawdown [%]",
    "Win Rate [%]", "Profit Factor", "# Trades",
]


def _embed_report(bt, key: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        html_path = Path(tmp_dir) / f"{key}.html"
        try:
            bt.plot(filename=str(html_path), open_browser=False)
            components.html(html_path.read_text(encoding="utf-8"), height=800, scrolling=True)
        except Exception as exc:
            st.info(f"Nie udało się wygenerować interaktywnego wykresu ({exc}) - statystyki poniżej są kompletne.")


def render():
    st.subheader("Backtest (walk-forward) sygnału ML")
    st.caption(
        "Symuluje handel na danych historycznych: prowizje, stop-loss/take-profit oparte o ATR, "
        "wielkość pozycji oparta o ryzyko, retrenowanie modelu co N świec na kroczącym oknie. "
        "Może potrwać od kilkudziesięciu sekund do kilku minut."
    )

    col1, col2, col3 = st.columns(3)
    ticker = col1.text_input("Ticker", value="AAPL", key="bt_ticker")
    period = col2.selectbox("Okres danych", ["6mo", "1y", "2y", "3y", "5y"], index=3, key="bt_period")
    interval = col3.selectbox("Interwał", ["1d", "1h", "1wk"], index=0, key="bt_interval")

    with st.expander("Parametry modelu i ryzyka", expanded=False):
        col4, col5, col6 = st.columns(3)
        horizon = col4.number_input("Horyzont (świece)", min_value=1, max_value=60, value=5, key="bt_horizon")
        atr_mult = col5.number_input("Próg ATR (atr_mult)", min_value=0.1, max_value=3.0, value=0.5, step=0.1, key="bt_atr_mult")
        min_confidence = col6.number_input("Min. pewność wejścia", min_value=0.34, max_value=0.95, value=0.40, step=0.01, key="bt_min_conf")

        col7, col8, col9 = st.columns(3)
        retrain_every = col7.number_input("Retrenuj co (świec)", min_value=5, max_value=100, value=20, key="bt_retrain_every")
        train_window = col8.number_input("Okno treningowe (świec)", min_value=120, max_value=1000, value=300, key="bt_train_window")
        risk_pct = col9.number_input("Ryzyko na transakcję", min_value=0.001, max_value=0.10, value=0.01, step=0.001, format="%.3f", key="bt_risk_pct")

        col10, col11, col12 = st.columns(3)
        sl_atr_mult = col10.number_input("SL (x ATR)", min_value=0.5, max_value=10.0, value=1.5, step=0.1, key="bt_sl")
        tp_atr_mult = col11.number_input("TP (x ATR)", min_value=0.5, max_value=10.0, value=2.5, step=0.1, key="bt_tp")
        cash = col12.number_input("Kapitał początkowy", min_value=1000, max_value=10_000_000, value=10_000, step=1000, key="bt_cash")

        col13, col14, col15 = st.columns(3)
        commission = col13.number_input("Prowizja", min_value=0.0, max_value=0.02, value=0.0007, step=0.0001, format="%.4f", key="bt_commission")
        max_drawdown_halt = col14.number_input("Kill-switch: max obsunięcie", min_value=0.05, max_value=0.90, value=0.25, step=0.05, key="bt_dd_halt")
        loss_streak_halt = col15.number_input("Kill-switch: seria strat", min_value=2, max_value=20, value=5, key="bt_loss_halt")

        col16, col17 = st.columns(2)
        sizing_mode = col16.selectbox(
            "Sposób wyznaczania SL/TP", ["quantile", "atr"], index=0, key="bt_sizing_mode",
            help="'quantile': z rozrzutu prognozy P10/P90 (z podłogą ATR); 'atr': stałe wielokrotności ATR",
        )
        benchmark = col17.text_input(
            "Indeks referencyjny (cechy reżimu rynku, puste = wyłącz)",
            value=DEFAULT_BENCHMARK, key="bt_benchmark",
        )

    compare_baselines = st.checkbox("Porównaj z prostymi strategiami bazowymi (turtle/sma/contrarian)", key="bt_compare")

    if not st.button("Uruchom backtest", type="primary", key="bt_run"):
        return

    try:
        with st.spinner(f"Pobieranie danych dla {ticker}..."):
            df = cached_fetch_ohlcv(ticker, period, interval)
    except (ValueError, ConnectionError) as exc:
        st.error(f"Błąd pobierania danych: {exc}")
        return

    try:
        quality = validate_ohlcv(df, min_rows=150)
        for warning in quality.warnings:
            st.warning(warning)
    except DataQualityError as exc:
        st.error(str(exc))
        return

    market_regime = None
    if benchmark:
        try:
            with st.spinner(f"Pobieranie indeksu referencyjnego {benchmark}..."):
                benchmark_df = cached_fetch_ohlcv(benchmark, period, interval)
            market_regime = derive_market_regime_features(benchmark_df)
        except (ValueError, ConnectionError) as exc:
            st.warning(f"Nie udało się pobrać indeksu referencyjnego ({exc}) - pomijam cechy reżimu rynku.")

    results = {}

    with st.spinner("Trenowanie i symulacja sygnału ML (może potrwać kilka minut)..."):
        bt_ml, stats_ml = run_backtest(
            df, horizon=horizon, atr_mult=atr_mult, retrain_every=retrain_every,
            train_window=train_window, min_confidence=min_confidence, risk_pct=risk_pct,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, sizing_mode=sizing_mode,
            max_drawdown_halt=max_drawdown_halt, loss_streak_halt=loss_streak_halt,
            market_regime=market_regime, cash=cash, commission=commission,
        )
    results["ML (ensemble)"] = stats_ml

    st.markdown("### Wyniki: sygnał ML")
    st.dataframe(stats_ml[SUMMARY_COLS].to_frame("Wartość"))
    _embed_report(bt_ml, "ml")

    trades = stats_ml["_trades"]
    if len(trades):
        with st.expander(f"Transakcje ({len(trades)})"):
            cols = [c for c in ["EntryTime", "ExitTime", "Size", "EntryPrice", "ExitPrice", "PnL", "ReturnPct"] if c in trades.columns]
            st.dataframe(trades[cols])

    if compare_baselines:
        baseline_params = {"turtle": {}, "sma": {}, "contrarian": {}}
        for name in BASELINES:
            with st.spinner(f"Strategia bazowa: {name}..."):
                bt_base, stats_base = run_baseline_backtest(
                    df, name, cash=cash, commission=commission, **baseline_params[name]
                )
            results[name] = stats_base
            with st.expander(f"Wyniki: {name}"):
                st.dataframe(stats_base[SUMMARY_COLS].to_frame("Wartość"))
                _embed_report(bt_base, name)

        st.markdown("### Porównanie strategii")
        summary = {label: {col: stats.get(col) for col in SUMMARY_COLS} for label, stats in results.items()}
        st.dataframe(pd.DataFrame(summary).T)
