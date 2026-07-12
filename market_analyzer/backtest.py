# ============================================================
# BACKTEST
#
# Measures how a ticker's price actually moved after a news item,
# and compares that to what the AI predicted.
#
# IMPORTANT: this must run as a SEPARATE, later pass - not right after
# ingestion. An article needs real elapsed time before a "4h later" or
# "24h later" price exists. Run this via `python main.py backtest`
# (e.g. on a schedule) once articles have aged past BACKTEST_MIN_AGE_HOURS.
# ============================================================
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from . import config, llm
from .db import get_db


def download_price(symbol, news_date, hours):
    try:
        start = news_date - timedelta(hours=3)
        end = news_date + timedelta(hours=hours + 24)
        data = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1h",
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None
        return data
    except Exception as e:
        print("[!] Yahoo error:", symbol, e)
        return None


def _pending_backtests():
    """news+assets rows old enough to backtest, that don't have a
    result yet for that (news_id, ticker) pair."""
    db = get_db()
    cur = db.cursor()
    cutoff = (datetime.now() - timedelta(hours=config.BACKTEST_MIN_AGE_HOURS)).isoformat()
    rows = cur.execute(
        """
        SELECT n.id, n.date, n.analysis, a.ticker
        FROM news n
        JOIN assets a ON a.news_id = n.id
        LEFT JOIN impact_backtest b ON b.news_id = n.id AND b.symbol = a.ticker
        WHERE n.date <= ? AND b.id IS NULL
        """,
        (cutoff,),
    ).fetchall()
    return rows


def run_backtest_for(news_id, symbol, analysis, news_date):
    data = download_price(symbol, news_date, 24)
    if data is None:
        return False

    try:
        # yfinance can return MultiIndex columns for a single symbol; normalize.
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        price_start = float(close.iloc[0])
        price_4h = float(close.iloc[min(4, len(close) - 1)])
        price_24h = float(close.iloc[-1])
    except Exception as e:
        print("[!] Price parse error:", symbol, e)
        return False

    return_4h = (price_4h - price_start) / price_start * 100
    return_24h = (price_24h - price_start) / price_start * 100

    direction = llm.get_prediction_direction(analysis)
    correct = 0
    if direction == "UP" and return_4h > 0:
        correct = 1
    elif direction == "DOWN" and return_4h < 0:
        correct = 1
    elif direction == "NEUTRAL" and abs(return_4h) < 0.5:
        correct = 1

    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO impact_backtest
        (news_id, symbol, predicted_direction, impact_score, price_at_news,
         price_after_4h, price_after_24h, return_4h, return_24h, was_correct, checked_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            news_id,
            symbol,
            direction,
            llm.get_impact_score(analysis),
            price_start,
            price_4h,
            price_24h,
            return_4h,
            return_24h,
            correct,
            datetime.now().isoformat(),
        ),
    )
    db.commit()
    return True


def run_pending_backtests():
    pending = _pending_backtests()
    if not pending:
        print("[i] Brak nowych backtestow do policzenia")
        return 0

    done = 0
    for news_id, news_date_str, analysis, symbol in pending:
        try:
            news_date = datetime.fromisoformat(news_date_str)
        except Exception:
            continue
        print(f"[~] Backtest {symbol} dla news #{news_id} ({news_date_str})")
        if run_backtest_for(news_id, symbol, analysis, news_date):
            done += 1

    print(f"[i] Policzono {done}/{len(pending)} backtestow")
    return done
