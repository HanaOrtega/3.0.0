"""Platforma GUI (Streamlit) spinająca cały projekt: sygnał ML, backtest i
sentyment z X w jednym interfejsie.

Uruchomienie:
    streamlit run app.py
"""

import streamlit as st

from ui import backtest_tab, info_tab, news_tab, signal_tab
from ui.common import init_session_state

st.set_page_config(page_title="Pattern ML", page_icon="📈", layout="wide")
init_session_state()

st.title("📈 Pattern ML — formacje świecowe, ML i sentyment z X")

tab_signal, tab_backtest, tab_news, tab_info = st.tabs(
    ["Sygnał ML", "Backtest", "Sentyment z X", "Informacje"]
)

with tab_signal:
    signal_tab.render()

with tab_backtest:
    backtest_tab.render()

with tab_news:
    news_tab.render()

with tab_info:
    info_tab.render()
