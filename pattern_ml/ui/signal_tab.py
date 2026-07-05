"""Zakładka GUI: sygnał ML (formacje świecowe + analiza techniczna + prognoza)."""

import numpy as np
import streamlit as st

from src.features import MAX_INDICATOR_LOOKBACK, build_feature_matrix
from src.model import predict_latest, predict_price_path, train_model, train_quantile_models
from src.plotting import plot_chart
from src.quality import DataQualityError, validate_ohlcv

from .common import cached_fetch_ohlcv, signal_badge


def render():
    st.subheader("Sygnał ML: formacje świecowe + analiza techniczna")

    col1, col2, col3 = st.columns(3)
    ticker = col1.text_input("Ticker", value="AAPL", key="signal_ticker")
    period = col2.selectbox("Okres danych", ["6mo", "1y", "2y", "5y"], index=2, key="signal_period")
    interval = col3.selectbox("Interwał", ["1d", "1h", "1wk"], index=0, key="signal_interval")

    col4, col5, col6 = st.columns(3)
    horizon = col4.number_input("Horyzont (świece)", min_value=1, max_value=60, value=5, key="signal_horizon")
    atr_mult = col5.number_input("Próg ATR (atr_mult)", min_value=0.1, max_value=3.0, value=0.5, step=0.1, key="signal_atr_mult")
    last_n = col6.slider("Świec na wykresie", min_value=50, max_value=400, value=150, key="signal_last_n")

    auto_confidence = st.checkbox(
        "Automatyczny próg pewności (maksymalizuje precyzję sygnałów, nie ogólną trafność)",
        value=True, key="signal_auto_conf",
    )
    manual_confidence = None
    if not auto_confidence:
        manual_confidence = st.slider("Ręczny próg pewności", min_value=0.34, max_value=0.90, value=0.40, step=0.01, key="signal_manual_conf")

    use_sentiment = False
    if st.session_state.get("sentiment_df") is not None:
        same_ticker = st.session_state.get("sentiment_ticker", "").upper() == ticker.upper()
        label = "Użyj sentymentu z zakładki 'Sentyment z X'"
        if not same_ticker:
            label += f" (uwaga: zebrany dla {st.session_state['sentiment_ticker']}, nie {ticker})"
        use_sentiment = st.checkbox(label, value=same_ticker, key="signal_use_sentiment")

    if not st.button("Uruchom analizę", type="primary", key="signal_run"):
        return

    try:
        with st.spinner(f"Pobieranie danych dla {ticker}..."):
            df = cached_fetch_ohlcv(ticker, period, interval)
    except (ValueError, ConnectionError) as exc:
        st.error(f"Błąd pobierania danych: {exc}")
        return

    st.caption(f"Pobrano {len(df)} świec: {df.index[0].date()} → {df.index[-1].date()}")

    try:
        quality = validate_ohlcv(df)
        for warning in quality.warnings:
            st.warning(warning)
    except DataQualityError as exc:
        st.error(str(exc))
        return

    sentiment_df = st.session_state["sentiment_df"] if use_sentiment else None

    with st.spinner("Liczenie wskaźników i trenowanie modelu ML..."):
        features, pat, X, y, y_reg, feature_cols = build_feature_matrix(
            df, horizon=horizon, atr_mult=atr_mult, sentiment=sentiment_df
        )
        if len(X) < 100:
            st.warning(f"Tylko {len(X)} próbek treningowych - rozważ dłuższy okres danych.")

        result = train_model(X, y, gap=horizon, embargo=MAX_INDICATOR_LOOKBACK)

        if manual_confidence is not None:
            min_confidence = manual_confidence
        elif not np.isnan(result.recommended_confidence_precision):
            min_confidence = result.recommended_confidence
        else:
            min_confidence = None

        signal, proba, as_of = predict_latest(result, features, feature_cols, min_confidence=min_confidence)

        quantile_models = train_quantile_models(X, y_reg)
        last_close = float(df["Close"].iloc[-1])
        quantile_prices = predict_price_path(quantile_models, features, feature_cols, last_close)

    st.markdown(signal_badge(signal, proba), unsafe_allow_html=True)
    st.caption(f"Na podstawie świecy z {as_of.date()}")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Cena obecna", f"{last_close:.2f}")
    col_b.metric(f"Prognoza mediana (+{horizon})", f"{quantile_prices[0.5]:.2f}",
                 delta=f"{(quantile_prices[0.5] / last_close - 1) * 100:.1f}%")
    col_c.metric("Zakres P10–P90", f"{quantile_prices[0.1]:.2f} – {quantile_prices[0.9]:.2f}")

    with st.spinner("Rysowanie wykresu..."):
        fig = plot_chart(
            df, features, pat, ticker, signal, proba,
            quantile_prices=quantile_prices, horizon=horizon, last_n=last_n,
        )
    st.pyplot(fig, clear_figure=True)

    with st.expander("Szczegóły modelu (walidacja krzyżowa, ważność cech)"):
        col_d, col_e = st.columns(2)
        col_d.metric("Trafność CV (walidacja krzyżowa szeregu czasowego)", f"{result.cv_accuracy * 100:.1f}%")
        if not np.isnan(result.recommended_confidence_precision):
            col_e.metric(
                "Zalecany próg pewności (precyzja sygnałów)",
                f"{result.recommended_confidence * 100:.0f}%",
                help=f"Szacowana precyzja sygnałów kierunkowych przy tym progu: "
                     f"{result.recommended_confidence_precision * 100:.0f}% (na danych walidacyjnych)",
            )
        st.text(result.report)
        st.bar_chart(result.feature_importances.head(10))
