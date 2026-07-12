# ============================================================
# BACKTEST - triple-barrier labeling + benchmark-relative alpha
#
# The original approach ("did the price simply go up 4h later?") has
# two well-known problems, both addressed here using standard
# quant-research techniques (Lopez de Prado, "Advances in Financial
# Machine Learning", and standard event-study methodology):
#
# 1. A fixed +/-0% threshold checked at a fixed 4h mark mislabels a
#    lot of noise as "correct"/"incorrect" - a stock that wiggles
#    +0.01% due to nothing in particular "confirms" a prediction just
#    by luck. Triple-barrier labeling instead sets an upper (profit)
#    and lower (stop) barrier sized to the ticker's OWN recent
#    volatility, plus a time-limit barrier, and asks "which wall did
#    the price hit first?" - a materially harder, more honest bar.
#
# 2. Raw return doesn't separate "this news moved the stock" from
#    "the whole market moved that day" (the difficulty/noise the user
#    asked to account for). We subtract the SPY return over the same
#    window to get alpha - the abnormal, stock-specific return.
#
# IMPORTANT: this must run as a SEPARATE, later pass - not right after
# ingestion. An article needs real elapsed time before a "4h later" or
# "24h later" price exists. Run via `python main.py backtest` (e.g. on
# a schedule) once articles have aged past BACKTEST_MIN_AGE_HOURS.
#
# Every resolved result feeds calibration.py, which is what lets the
# system "learn from its mistakes": sources/event types/sectors that
# have historically led the AI astray get down-weighted in future
# recommendations, and ones that have proven reliable get up-weighted.
# ============================================================
import math
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from . import calibration, config, llm
from .db import get_db

BENCHMARK_SYMBOL = "SPY"
BARRIER_SIGMA_MULTIPLIER = 1.5


def _normalize_close(data):
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close


def download_price(symbol, news_date, hours):
    try:
        start = news_date - timedelta(hours=3)
        end = news_date + timedelta(hours=hours + 24)
        data = yf.download(symbol, start=start, end=end, interval="1h", progress=False, auto_adjust=True)
        if data.empty:
            return None
        return data
    except Exception as e:
        print("[!] Yahoo error:", symbol, e)
        return None


def _recent_daily_volatility(symbol, news_date, lookback_days=20):
    """Std of daily returns in the ~lookback_days before the news, used
    to size the triple-barrier width to how volatile this ticker
    normally is (a 2% move means something different for a utility
    stock than for a small-cap biotech)."""
    try:
        start = news_date - timedelta(days=lookback_days + 5)
        data = yf.download(symbol, start=start, end=news_date, interval="1d", progress=False, auto_adjust=True)
        close = _normalize_close(data)
        if len(close) < 5:
            return 0.02  # fallback: assume 2% daily vol
        returns = close.pct_change().dropna()
        return float(returns.std()) or 0.02
    except Exception:
        return 0.02


def _triple_barrier_label(close_series, daily_vol, horizon_hours):
    """Walk the hourly price path forward from close_series[0] and
    return (label, barrier_hit, hit_index) where label is +1/-1/0 for
    upper/lower/timeout barrier, sized to daily_vol scaled to the
    elapsed horizon (variance grows ~linearly with time)."""
    price_start = float(close_series.iloc[0])
    max_idx = min(len(close_series) - 1, horizon_hours)

    for i in range(1, max_idx + 1):
        hours_elapsed = i
        scaled_vol = daily_vol * math.sqrt(hours_elapsed / 24.0)
        barrier = max(BARRIER_SIGMA_MULTIPLIER * scaled_vol, 0.002)  # floor: 0.2%
        price = float(close_series.iloc[i])
        change = (price - price_start) / price_start

        if change >= barrier:
            return 1, "upper", i
        if change <= -barrier:
            return -1, "lower", i

    return 0, "timeout", max_idx


def _pending_backtests():
    db = get_db()
    cur = db.cursor()
    cutoff = (datetime.now() - timedelta(hours=config.BACKTEST_MIN_AGE_HOURS)).isoformat()
    rows = cur.execute(
        """
        SELECT n.id, n.date, n.analysis, n.category, a.ticker, a.sector
        FROM news n
        JOIN assets a ON a.news_id = n.id
        LEFT JOIN impact_backtest b ON b.news_id = n.id AND b.symbol = a.ticker
        WHERE n.date <= ? AND b.id IS NULL
        """,
        (cutoff,),
    ).fetchall()
    return rows


def run_backtest_for(news_id, symbol, analysis, news_date, source=None, sector=None, benchmark_cache=None):
    data = download_price(symbol, news_date, 24)
    if data is None:
        return False

    try:
        close = _normalize_close(data)
        if len(close) < 2:
            return False
        price_start = float(close.iloc[0])
        price_4h = float(close.iloc[min(4, len(close) - 1)])
        price_24h = float(close.iloc[-1])
    except Exception as e:
        print("[!] Price parse error:", symbol, e)
        return False

    return_4h = (price_4h - price_start) / price_start * 100
    return_24h = (price_24h - price_start) / price_start * 100

    # Benchmark (alpha = stock return - market return over the same window)
    benchmark_return_4h = benchmark_return_24h = None
    alpha_4h = alpha_24h = None
    bench_data = None
    if benchmark_cache is not None:
        cache_key = news_date.strftime("%Y-%m-%d-%H")
        bench_data = benchmark_cache.get(cache_key)
        if bench_data is None:
            bench_data = download_price(BENCHMARK_SYMBOL, news_date, 24)
            benchmark_cache[cache_key] = bench_data
    else:
        bench_data = download_price(BENCHMARK_SYMBOL, news_date, 24)

    if bench_data is not None:
        try:
            bench_close = _normalize_close(bench_data)
            if len(bench_close) >= 2:
                b_start = float(bench_close.iloc[0])
                b_4h = float(bench_close.iloc[min(4, len(bench_close) - 1)])
                b_24h = float(bench_close.iloc[-1])
                benchmark_return_4h = (b_4h - b_start) / b_start * 100
                benchmark_return_24h = (b_24h - b_start) / b_start * 100
                alpha_4h = return_4h - benchmark_return_4h
                alpha_24h = return_24h - benchmark_return_24h
        except Exception:
            pass

    daily_vol = _recent_daily_volatility(symbol, news_date)
    horizon_hours = min(int(llm.get_time_horizon_hours(analysis)), len(close) - 1) or len(close) - 1
    label, barrier_hit, _ = _triple_barrier_label(close, daily_vol, max(horizon_hours, 1))

    direction = llm.get_prediction_direction(analysis)
    correct = 0
    if direction == "UP" and label == 1:
        correct = 1
    elif direction == "DOWN" and label == -1:
        correct = 1
    elif direction == "NEUTRAL" and label == 0:
        correct = 1

    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO impact_backtest
        (news_id, symbol, predicted_direction, impact_score, price_at_news,
         price_after_4h, price_after_24h, return_4h, return_24h, was_correct, checked_at,
         benchmark_symbol, benchmark_return_4h, benchmark_return_24h, alpha_4h, alpha_24h,
         barrier_hit, label)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            news_id, symbol, direction, llm.get_impact_score(analysis), price_start,
            price_4h, price_24h, return_4h, return_24h, correct, datetime.now().isoformat(),
            BENCHMARK_SYMBOL, benchmark_return_4h, benchmark_return_24h, alpha_4h, alpha_24h,
            barrier_hit, label,
        ),
    )
    db.commit()

    event_type = llm.get_event_type(analysis)
    calibration.update_from_backtest(source=source, event_type=event_type, sector=sector, correct=bool(correct))

    return True


def run_pending_backtests():
    pending = _pending_backtests()
    if not pending:
        print("[i] Brak nowych backtestow do policzenia")
        return 0

    benchmark_cache = {}
    done = 0
    for news_id, news_date_str, analysis, source, symbol, sector in pending:
        try:
            news_date = datetime.fromisoformat(news_date_str)
        except Exception:
            continue
        print(f"[~] Backtest {symbol} dla news #{news_id} ({news_date_str})")
        if run_backtest_for(news_id, symbol, analysis, news_date, source=source, sector=sector, benchmark_cache=benchmark_cache):
            done += 1

    print(f"[i] Policzono {done}/{len(pending)} backtestow")
    return done
