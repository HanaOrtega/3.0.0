# ============================================================
# RECOMMENDATION ENGINE
#
# Aggregates recent AI-scored news per ticker into a single
# BUY / SELL / HOLD call with a confidence score and a written
# rationale (which headlines drove it, and how the AI has
# historically performed on that ticker per the backtest table).
# ============================================================
import math
from datetime import datetime, timedelta

from . import config, llm
from .db import get_db

_SENTIMENT_VALUE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_DIRECTION_VALUE = {"UP": 1.0, "NEUTRAL": 0.0, "DOWN": -1.0}


def _recency_weight(news_date, now):
    age_hours = max((now - news_date).total_seconds() / 3600.0, 0)
    return 0.5 ** (age_hours / config.RECO_HALF_LIFE_HOURS)


def _fetch_recent_signals(lookback_days):
    db = get_db()
    cur = db.cursor()
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    rows = cur.execute(
        """
        SELECT a.ticker, a.company, a.sector, n.id, n.date, n.title, n.analysis
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
        lines.append(
            f"- [{c['direction']}/{c['sentiment']}, impact {c['impact']:.1f}] {c['title']}"
        )
    if backtest_n:
        lines.append(
            f"\nHistoryczna skutecznosc AI dla {ticker}: {backtest_acc * 100:.0f}% "
            f"(na podstawie {backtest_n} zweryfikowanych newsow)."
        )
    else:
        lines.append(f"\nBrak jeszcze historycznych danych backtest dla {ticker}.")
    return "\n".join(lines)


def generate_recommendations(lookback_days=None, save=True):
    lookback_days = lookback_days or config.RECO_LOOKBACK_DAYS
    rows = _fetch_recent_signals(lookback_days)

    by_ticker = {}
    for ticker, company, sector, news_id, date_str, title, analysis in rows:
        try:
            news_date = datetime.fromisoformat(date_str)
        except Exception:
            continue
        parsed = llm.parse_analysis(analysis)
        sentiment = parsed.get("sentiment", "neutral")
        direction = parsed.get("direction", "NEUTRAL")
        impact = float(parsed.get("impact_score", 0) or 0)
        model_conf = float(parsed.get("confidence", 0) or 0) or 0.5

        base_signal = (
            _SENTIMENT_VALUE.get(sentiment, 0.0) + _DIRECTION_VALUE.get(direction, 0.0)
        ) / 2.0
        weight = max(impact, 0.1) * model_conf * _recency_weight(news_date, datetime.now())

        entry = by_ticker.setdefault(
            ticker,
            {"company": set(), "sector": sector, "signals": [], "contributions": []},
        )
        entry["company"].add(company)
        entry["signals"].append((base_signal, weight))
        entry["contributions"].append(
            {
                "title": title,
                "direction": direction,
                "sentiment": sentiment,
                "impact": impact,
                "weight": weight,
            }
        )

    results = []
    now_str = datetime.now().isoformat()

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
             news_count, backtest_accuracy, rationale, generated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
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
            ),
        )
    db.commit()


def print_recommendations(results):
    print(f"\n{config.DISCLAIMER}\n")
    for r in results:
        print(
            f"{r['action']:5s} {r['ticker']:6s} ({r['company']}) "
            f"score={r['score']:+.2f} confidence={r['confidence']:.2f} "
            f"news={r['news_count']}"
        )
        print(r["rationale"])
        print("-" * 60)
