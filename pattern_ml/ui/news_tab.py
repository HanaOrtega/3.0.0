"""Zakładka GUI: świeże wzmianki o instrumencie na X (Twitter) -> sentyment do modelu ML."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from playwright.sync_api import sync_playwright

from src.sentiment import aggregate_x_items
from src.x_auth import STORAGE_STATE_PATH, LoginError, ensure_logged_in
from src.x_scraper import fetch_fresh_mentions

CHROMIUM_PATH = "/opt/pw-browsers/chromium"
DEFAULT_ACCOUNTS = "DeItaone,unusual_whales,FirstSquawk,Reuters,business"


def render():
    st.subheader("Świeże wzmianki o instrumencie na X (Twitter)")
    st.caption(
        "Loguje się do X i zbiera WYŁĄCZNIE świeże posty (zakładka 'Najnowsze', nie 'Najlepsze'), "
        "żeby uniknąć starych/archiwalnych wzmianek. Wynik można od razu wykorzystać jako dodatkowe "
        "cechy sentymentu w zakładce 'Sygnał ML'."
    )

    has_creds = bool(os.environ.get("X_USERNAME")) and bool(os.environ.get("X_PASSWORD"))
    if not has_creds:
        st.warning(
            "Brak zmiennych środowiskowych X_USERNAME / X_PASSWORD - ustaw je PRZED uruchomieniem "
            "`streamlit run app.py` (np. `export X_USERNAME=... X_PASSWORD=...`). Hasło nigdy nie "
            "jest wpisywane w tym interfejsie."
        )

    col1, col2 = st.columns(2)
    ticker = col1.text_input("Ticker", value="AAPL", key="news_ticker")
    company = col2.text_input("Pełna nazwa spółki (opcjonalnie)", value="", key="news_company")

    accounts_raw = st.text_input(
        "Konta finansowe/newsowe monitorowane (bez @, po przecinku)",
        value=DEFAULT_ACCOUNTS, key="news_accounts",
    )

    col3, col4 = st.columns(2)
    hours = col3.number_input("Maks. wiek posta (godziny)", min_value=1, max_value=168, value=24, key="news_hours")
    max_scrolls = col4.number_input("Doładowań wyników wyszukiwania", min_value=1, max_value=15, value=5, key="news_max_scrolls")

    if not st.button("Szukaj świeżych wzmianek", type="primary", key="news_run", disabled=not has_creds):
        return

    accounts = [a.strip() for a in accounts_raw.split(",") if a.strip()]

    try:
        with st.spinner("Logowanie do X i wyszukiwanie świeżych wzmianek (może potrwać minutę)..."):
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, executable_path=CHROMIUM_PATH)
                context = browser.new_context(
                    storage_state=str(STORAGE_STATE_PATH) if STORAGE_STATE_PATH.exists() else None,
                    viewport={"width": 1280, "height": 900},
                )
                page = context.new_page()
                ensure_logged_in(context, page)
                items = fetch_fresh_mentions(
                    page, ticker, company or None, accounts, hours, max_scrolls=max_scrolls
                )
                browser.close()
    except LoginError as exc:
        st.error(f"Błąd logowania: {exc}")
        return

    if not items:
        st.info("Brak świeżych wzmianek w podanym oknie czasowym.")
        st.session_state["sentiment_df"] = None
        return

    df_items = pd.DataFrame(items)
    st.success(f"Znaleziono {len(items)} świeżych wzmianek.")
    st.dataframe(
        df_items[["published_at", "author_handle", "text", "likes", "retweets", "replies"]],
        use_container_width=True,
    )

    sentiment_df = aggregate_x_items(items)
    st.session_state["sentiment_df"] = sentiment_df
    st.session_state["sentiment_ticker"] = ticker
    st.session_state["sentiment_meta"] = {
        "count": len(items),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    st.info(
        f"Sentyment zagregowany i zapamiętany dla {ticker} - w zakładce 'Sygnał ML' zaznacz "
        "'Użyj sentymentu z zakładki Sentyment z X', żeby dołączyć go do cech modelu."
    )

    payload = {
        "ticker": ticker.upper(), "company": company or None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freshness_hours": hours, "monitored_accounts": accounts,
        "count": len(items), "items": items,
    }
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"news_{ticker.upper()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    st.caption(f"Wynik zapisano też do pliku: {out_path}")
