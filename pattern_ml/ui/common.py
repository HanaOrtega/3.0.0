"""Współdzielone pomocniki GUI: cache'owane pobieranie danych i stan sesji."""

import streamlit as st

from src.data import fetch_ohlcv

SIGNAL_COLORS = {"LONG": "#1a9850", "SHORT": "#d73027", "NEUTRALNY": "#999999"}


@st.cache_data(show_spinner=False, ttl=300)
def cached_fetch_ohlcv(ticker: str, period: str, interval: str):
    """Pobiera OHLCV z yfinance, cache'owane na 5 minut per (ticker, period, interval)."""
    return fetch_ohlcv(ticker, period=period, interval=interval)


def init_session_state():
    defaults = {
        "sentiment_df": None,
        "sentiment_ticker": None,
        "sentiment_meta": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def signal_badge(signal: str, proba: dict) -> str:
    """Zwraca HTML znaczka sygnału (LONG/SHORT/NEUTRALNY) do st.markdown(unsafe_allow_html=True)."""
    color = SIGNAL_COLORS.get(signal, "#333333")
    proba_txt = " &nbsp;|&nbsp; ".join(f"{k}: {v * 100:.1f}%" for k, v in proba.items())
    return (
        f'<div style="background-color:{color};color:white;padding:0.75rem 1rem;'
        f'border-radius:0.5rem;font-weight:600;font-size:1.1rem;">'
        f"Sygnał ML: {signal}<br>"
        f'<span style="font-weight:400;font-size:0.9rem;">{proba_txt}</span></div>'
    )
