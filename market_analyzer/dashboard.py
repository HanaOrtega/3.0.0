# ============================================================
# STREAMLIT DASHBOARD
# ============================================================
import sqlite3

import pandas as pd

from . import config


def _load_tables():
    db_local = sqlite3.connect(config.DB_FILE)
    news = pd.read_sql("SELECT * FROM news ORDER BY date DESC", db_local)
    assets = pd.read_sql("SELECT * FROM assets", db_local)
    backtest = pd.read_sql("SELECT * FROM impact_backtest", db_local)
    recos = pd.read_sql("SELECT * FROM recommendations ORDER BY generated_at DESC", db_local)
    digests = pd.read_sql("SELECT * FROM digests ORDER BY created_at DESC", db_local)
    db_local.close()
    return news, assets, backtest, recos, digests


def dashboard():
    import plotly.express as px
    import streamlit as st

    st.set_page_config(page_title="AI Market Analyzer", layout="wide")
    st.title("AI Market News Analyzer")
    st.caption(config.DISCLAIMER)

    news, assets, backtest, recos, digests = _load_tables()

    tab_overview, tab_reco, tab_digest, tab_news, tab_backtest = st.tabs(
        ["Przeglad", "Rekomendacje Kup/Sprzedaj", "Raport wplywu", "News", "Skutecznosc AI"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("News", len(news))
        c2.metric("Tickery", assets["ticker"].nunique() if len(assets) else 0)
        accuracy = backtest["was_correct"].mean() * 100 if len(backtest) else 0
        c3.metric("Skutecznosc AI", f"{accuracy:.1f}%")
        avg_move = backtest["return_4h"].mean() if len(backtest) else 0
        c4.metric("Sredni ruch 4h", f"{avg_move:.2f}%")

        if len(recos):
            latest_gen = recos["generated_at"].max()
            latest = recos[recos["generated_at"] == latest_gen]
            b = (latest["action"] == "BUY").sum()
            s = (latest["action"] == "SELL").sum()
            h = (latest["action"] == "HOLD").sum()
            st.subheader("Ostatnie rekomendacje - podsumowanie")
            c1, c2, c3 = st.columns(3)
            c1.metric("KUPUJ", b)
            c2.metric("SPRZEDAJ", s)
            c3.metric("TRZYMAJ", h)

    with tab_reco:
        st.subheader("Aktualne rekomendacje Kup / Sprzedaj / Trzymaj")
        if len(recos):
            latest_gen = recos["generated_at"].max()
            latest = recos[recos["generated_at"] == latest_gen].copy()
            latest = latest.sort_values(
                by="score", key=lambda s: s.abs(), ascending=False
            )

            st.dataframe(
                latest[
                    ["action", "ticker", "company", "sector", "score", "confidence",
                     "backtest_accuracy", "news_count"]
                ],
                use_container_width=True,
            )

            st.subheader("Uzasadnienie")
            pick = st.selectbox("Wybierz ticker", latest["ticker"].tolist())
            if pick:
                row = latest[latest["ticker"] == pick].iloc[0]
                st.markdown(f"**{row['action']} {row['ticker']}** ({row['company']})")
                st.text(row["rationale"])
        else:
            st.info("Brak wygenerowanych rekomendacji. Uruchom `python main.py recommend`.")

    with tab_digest:
        st.subheader("Ostatnie raporty wplywu")
        if len(digests):
            for _, row in digests.head(5).iterrows():
                with st.expander(f"{row['period']} — {row['date']}"):
                    st.markdown(row["summary"])
        else:
            st.info("Brak raportow. Uruchom `python main.py digest`.")

    with tab_news:
        st.subheader("Ostatnie wiadomosci")
        if len(news):
            st.dataframe(
                news[["date", "title", "category"]].head(50),
                use_container_width=True,
            )
            pick = st.selectbox("Podgląd analizy", news["title"].head(50).tolist())
            if pick:
                row = news[news["title"] == pick].iloc[0]
                st.json(row["analysis"])
        else:
            st.info("Brak newsow. Uruchom `python main.py scan`.")

    with tab_backtest:
        st.subheader("Ruch cen po newsach")
        if len(backtest):
            fig = px.histogram(backtest, x="return_4h", nbins=40)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Ranking tickerow")
            ranking = (
                backtest.groupby("symbol")
                .agg(
                    liczba=("symbol", "count"),
                    skutecznosc=("was_correct", "mean"),
                    ruch_4h=("return_4h", "mean"),
                    ruch_24h=("return_24h", "mean"),
                )
                .reset_index()
            )
            ranking["skutecznosc"] *= 100
            st.dataframe(
                ranking.sort_values("skutecznosc", ascending=False),
                use_container_width=True,
            )
        else:
            st.info("Brak wynikow backtestu. Uruchom `python main.py backtest`.")


if __name__ == "__main__":
    dashboard()
