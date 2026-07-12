# ============================================================
# RECOMMENDATION ENGINE
#
# Aggregates recent AI-scored news per ticker into a single
# BUY / SELL / HOLD call with a confidence score and a written
# rationale. Three "learning" ingredients feed in on top of the raw
# per-article signal:
#
#   - calibration.py: a Beta-Bernoulli trust score per news source,
#     event type and sector, updated every time backtest.py resolves
#     a prediction - sources/event-types that have been wrong before
#     get down-weighted automatically.
#   - dedupe.py: independent corroboration (several outlets covering
#     the same event) is rewarded, but with diminishing returns, so a
#     single story republished five times doesn't out-vote one
#     genuinely distinct article.
#   - finbert.py (optional): a finance-tuned sentiment model blended
#     with the LLM's own sentiment, since generic LLMs misread finance
#     jargon fairly often.
#
# A separate "today" score is also tracked (last 24h only, higher bar)
# so the system can answer "what's worth acting on TODAY" distinctly
# from its slower-moving multi-day view of a ticker.
# ============================================================
import math
from datetime import datetime, timedelta

from . import calibration, config, dedupe, llm
from .db import get_db

_DIRECTION_VALUE = {"UP": 1.0, "NEUTRAL": 0.0, "DOWN": -1.0}

TODAY_WINDOW_HOURS = 24
TODAY_SCORE_THRESHOLD = 0.5
TODAY_CONFIDENCE_THRESHOLD = 0.6


def _recency_weight(news_date, now):
    age_hours = max((now - news_date).total_seconds() / 3600.0, 0)
    return 0.5 ** (age_hours / config.RECO_HALF_LIFE_HOURS)


def _fetch_recent_signals(lookback_days):
    db = get_db()
    cur = db.cursor()
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    rows = cur.execute(
        """
        SELECT a.ticker, a.company, a.sector, a.event_cluster,
               n.id, n.date, n.title, n.content, n.analysis, n.category
        FROM assets a
        JOIN news n ON n.id = a.news_id
        WHERE n.date >= ?
        ORDER BY a.ticker, n.date DESC
        """,
        (cutoff,),
    ).fetchall()
    return rows


def _backtest_accuracy(ticker):
    db = get_db()
    cur = db.cursor()
    row = cur.execute(
        "SELECT AVG(was_correct), COUNT(*) FROM impact_backtest WHERE symbol=?",
        (ticker,),
    ).fetchone()
    if row and row[1]:
        return float(row[0]), int(row[1])
    return None, 0


def _action_for_score(score, news_count):
    if news_count < config.RECO_MIN_NEWS:
        return "HOLD"
    if score >= config.RECO_BUY_THRESHOLD:
        return "BUY"
    if score <= config.RECO_SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def _build_rationale(ticker, contributions, backtest_acc, backtest_n):
    contributions = sorted(contributions, key=lambda c: c["weight"], reverse=True)
    lines = []
    for c in contributions[:5]:
        corrob = f", {c['corroboration']}x zrodel" if c["corroboration"] > 1 else ""
        lines.append(
            f"- [{c['direction']}/{c['sentiment']}, impact {c['impact']:.1f}{corrob}] {c['title']}"
        )
    if backtest_n:
        lines.append(
            f"\nHistoryczna skutecznosc AI dla {ticker}: {backtest_acc * 100:.0f}% "
            f"(na podstawie {backtest_n} zweryfikowanych newsow)."
        )
    else:
        lines.append(f"\nBrak jeszcze historycznych danych backtest dla {ticker}.")
    return "\n".join(lines)


def generate_recommendations(lookback_days=None, save=True, use_finbert=True):
    lookback_days = lookback_days or config.RECO_LOOKBACK_DAYS
    rows = _fetch_recent_signals(lookback_days)
    now = datetime.now()

    by_ticker = {}
    for ticker, company, sector, cluster, news_id, date_str, title, content, analysis, source in rows:
        try:
            news_date = datetime.fromisoformat(date_str)
        except Exception:
            continue

        parsed = llm.parse_analysis(analysis)
        sentiment = parsed.get("sentiment", "neutral")
        direction = parsed.get("direction", "NEUTRAL")
        impact = float(parsed.get("impact_score", 0) or 0)
        model_conf = float(parsed.get("confidence", 0) or 0) or 0.5
        event_type = parsed.get("event_type", "other")

        if use_finbert:
            sentiment_value, _ = llm.blended_sentiment_value(content or "", analysis)
        else:
            sentiment_value = llm.SENTIMENT_VALUE.get(sentiment, 0.0)

        base_signal = (sentiment_value + _DIRECTION_VALUE.get(direction, 0.0)) / 2.0

        corrob_count = dedupe.corroboration_count(ticker, cluster)
        trust = calibration.get_weight(source=source, event_type=event_type, sector=sector)

        weight = (
            max(impact, 0.1)
            * model_conf
            * _recency_weight(news_date, now)
            * trust
            * dedupe.corroboration_weight(corrob_count)
        )

        is_today = (now - news_date).total_seconds() / 3600.0 <= TODAY_WINDOW_HOURS

        entry = by_ticker.setdefault(
            ticker,
            {
                "company": set(), "sector": sector,
                "signals": [], "contributions": [],
                "today_signals": [],
            },
        )
        entry["company"].add(company)
        entry["signals"].append((base_signal, weight))
        entry["contributions"].append(
            {
                "title": title, "direction": direction, "sentiment": sentiment,
                "impact": impact, "weight": weight, "corroboration": corrob_count,
            }
        )
        if is_today:
            entry["today_signals"].append((base_signal, weight))

    results = []
    now_str = now.isoformat()

    for ticker, entry in by_ticker.items():
        signals = entry["signals"]
        total_weight = sum(w for _, w in signals)
        if total_weight <= 0:
            continue
        weighted_score = sum(s * w for s, w in signals) / total_weight

        mean_signal = sum(s for s, _ in signals) / len(signals)
        variance = sum((s - mean_signal) ** 2 for s, _ in signals) / len(signals)
        agreement = 1.0 - min(math.sqrt(variance), 1.0)

        backtest_acc, backtest_n = _backtest_accuracy(ticker)
        backtest_component = backtest_acc if backtest_acc is not None else 0.5

        news_count = len(signals)
        volume_component = min(news_count / 5.0, 1.0)

        confidence = round(
            (0.4 * agreement + 0.3 * backtest_component + 0.3 * volume_component), 3
        )

        action = _action_for_score(weighted_score, news_count)
        rationale = _build_rationale(
            ticker, entry["contributions"], backtest_acc or 0.0, backtest_n
        )

        today_signals = entry["today_signals"]
        today_weight = sum(w for _, w in today_signals)
        score_today = (sum(s * w for s, w in today_signals) / today_weight) if today_weight > 0 else None
        high_conviction_today = (
            score_today is not None
            and len(today_signals) >= 1
            and abs(score_today) >= TODAY_SCORE_THRESHOLD
            and confidence >= TODAY_CONFIDENCE_THRESHOLD
        )

        result = {
            "ticker": ticker,
            "company": ", ".join(sorted(entry["company"])),
            "sector": entry["sector"],
            "action": action,
            "score": round(weighted_score, 3),
            "confidence": confidence,
            "avg_sentiment": round(mean_signal, 3),
            "news_count": news_count,
            "backtest_accuracy": backtest_acc,
            "rationale": rationale,
            "generated_at": now_str,
            "score_today": round(score_today, 3) if score_today is not None else None,
            "news_count_today": len(today_signals),
            "high_conviction_today": high_conviction_today,
        }
        results.append(result)

    results.sort(key=lambda r: abs(r["score"]) * r["confidence"], reverse=True)

    if save and results:
        _save_recommendations(results)

    return results


def _save_recommendations(results):
    db = get_db()
    cur = db.cursor()
    for r in results:
        cur.execute(
            """
            INSERT INTO recommendations
            (ticker, company, sector, action, score, confidence, avg_sentiment,
             news_count, backtest_accuracy, rationale, generated_at,
             score_today, news_count_today, high_conviction_today)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                r["ticker"],
                r["company"],
                r["sector"],
                r["action"],
                r["score"],
                r["confidence"],
                r["avg_sentiment"],
                r["news_count"],
                r["backtest_accuracy"],
                r["rationale"],
                r["generated_at"],
                r["score_today"],
                r["news_count_today"],
                int(r["high_conviction_today"]),
            ),
        )
    db.commit()


def today_highlights(results):
    return [r for r in results if r["high_conviction_today"] and r["action"] != "HOLD"]


def print_recommendations(results):
    print(f"\n{config.DISCLAIMER}\n")
    for r in results:
        today_flag = " [DZIS]" if r["high_conviction_today"] else ""
        print(
            f"{r['action']:5s} {r['ticker']:6s} ({r['company']}){today_flag} "
            f"score={r['score']:+.2f} confidence={r['confidence']:.2f} "
            f"news={r['news_count']}"
        )
        print(r["rationale"])
        print("-" * 60)
