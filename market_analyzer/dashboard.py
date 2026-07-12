# ============================================================
# STREAMLIT DASHBOARD
# ============================================================
import sqlite3

import pandas as pd

from . import calibration, config


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

    tab_overview, tab_reco, tab_digest, tab_news, tab_backtest, tab_learning = st.tabs(
        ["Przeglad", "Rekomendacje Kup/Sprzedaj", "Raport wplywu", "News",
         "Skutecznosc AI", "Uczenie sie / Kalibracja"]
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

            today_hits = latest[
                (latest["high_conviction_today"] == 1) & (latest["action"] != "HOLD")
            ]
            if len(today_hits):
                st.warning(
                    "**Dzis warto zwrocic uwage:** "
                    + ", ".join(f"{row['action']} {row['ticker']}" for _, row in today_hits.iterrows())
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
            agg_kwargs = dict(
                liczba=("symbol", "count"),
                skutecznosc=("was_correct", "mean"),
                ruch_4h=("return_4h", "mean"),
                ruch_24h=("return_24h", "mean"),
            )
            if "alpha_4h" in backtest.columns:
                agg_kwargs["alpha_4h"] = ("alpha_4h", "mean")
                agg_kwargs["alpha_24h"] = ("alpha_24h", "mean")
            ranking = backtest.groupby("symbol").agg(**agg_kwargs).reset_index()
            ranking["skutecznosc"] *= 100
            st.caption("alpha = ruch tickera minus ruch SPY w tym samym oknie (efekt oczyszczony z ruchu calego rynku)")
            st.dataframe(
                ranking.sort_values("skutecznosc", ascending=False),
                use_container_width=True,
            )
        else:
            st.info("Brak wynikow backtestu. Uruchom `python main.py backtest`.")

    with tab_learning:
        st.subheader("Jak system sie uczy")
        st.caption(
            "Kazdy rozwiazany backtest (triple-barrier vs SPY) aktualizuje te wagi. "
            "Zrodla/typy zdarzen z niska skutecznoscia sa automatycznie obnizane w przyszlych rekomendacjach."
        )

        source_cal = calibration.snapshot("source")
        event_cal = calibration.snapshot("event_type")
        sector_cal = calibration.snapshot("sector")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Zrodla RSS**")
            if source_cal:
                st.dataframe(pd.DataFrame(source_cal), use_container_width=True)
            else:
                st.info("Brak jeszcze danych.")
        with col2:
            st.markdown("**Typy zdarzen**")
            if event_cal:
                st.dataframe(pd.DataFrame(event_cal), use_container_width=True)
            else:
                st.info("Brak jeszcze danych.")
        with col3:
            st.markdown("**Sektory**")
            if sector_cal:
                st.dataframe(pd.DataFrame(sector_cal), use_container_width=True)
            else:
                st.info("Brak jeszcze danych.")


if __name__ == "__main__":
    dashboard()
