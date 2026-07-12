# ============================================================
# IMPACT DIGEST
#
# Turns the raw news/analysis/recommendation data into a short
# written "what happened and what it means" summary - grouped by
# RSS category and by sector, highlighting the highest-impact
# stories and any BUY/SELL calls.
# ============================================================
import json
import os
from datetime import datetime, timedelta

from . import config, llm, recommend
from .db import get_db

_PERIOD_HOURS = {"daily": 24, "weekly": 24 * 7}


def _fetch_period_news(hours):
    db = get_db()
    cur = db.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    rows = cur.execute(
        "SELECT id, date, title, url, category, analysis FROM news WHERE date >= ? ORDER BY date DESC",
        (cutoff,),
    ).fetchall()
    return rows


def _sector_breakdown(hours):
    db = get_db()
    cur = db.cursor()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    rows = cur.execute(
        """
        SELECT a.sector, a.ticker, n.analysis
        FROM assets a
        JOIN news n ON n.id = a.news_id
        WHERE n.date >= ? AND a.sector != ''
        """,
        (cutoff,),
    ).fetchall()

    sectors = {}
    for sector, ticker, analysis in rows:
        impact = llm.get_impact_score(analysis)
        entry = sectors.setdefault(sector, {"count": 0, "impact_sum": 0.0, "tickers": set()})
        entry["count"] += 1
        entry["impact_sum"] += impact
        entry["tickers"].add(ticker)

    breakdown = []
    for sector, e in sectors.items():
        breakdown.append(
            {
                "sector": sector,
                "count": e["count"],
                "avg_impact": e["impact_sum"] / e["count"] if e["count"] else 0,
                "tickers": sorted(e["tickers"]),
            }
        )
    breakdown.sort(key=lambda x: x["avg_impact"] * x["count"], reverse=True)
    return breakdown


def build_digest(period="daily"):
    hours = _PERIOD_HOURS.get(period, 24)
    news_rows = _fetch_period_news(hours)

    scored = []
    by_category = {}
    for news_id, date, title, url, category, analysis in news_rows:
        parsed = llm.parse_analysis(analysis)
        impact = float(parsed.get("impact_score", 0) or 0)
        scored.append(
            {
                "id": news_id,
                "date": date,
                "title": title,
                "url": url,
                "category": category,
                "impact": impact,
                "sentiment": parsed.get("sentiment", "neutral"),
                "direction": parsed.get("direction", "NEUTRAL"),
                "summary": parsed.get("summary", ""),
            }
        )
        cat = by_category.setdefault(category, {"count": 0, "impact_sum": 0.0})
        cat["count"] += 1
        cat["impact_sum"] += impact

    top_stories = sorted(scored, key=lambda s: s["impact"], reverse=True)[:8]
    sector_breakdown = _sector_breakdown(hours)

    recos = recommend.generate_recommendations(save=False)
    buy_calls = [r for r in recos if r["action"] == "BUY"]
    sell_calls = [r for r in recos if r["action"] == "SELL"]

    md = _render_markdown(
        period, news_rows, by_category, top_stories, sector_breakdown, buy_calls, sell_calls
    )

    top_movers = {
        "buy": [{"ticker": r["ticker"], "score": r["score"]} for r in buy_calls[:10]],
        "sell": [{"ticker": r["ticker"], "score": r["score"]} for r in sell_calls[:10]],
    }

    _save_digest(period, md, top_movers)
    _write_report_file(period, md)

    return md


def _render_markdown(period, news_rows, by_category, top_stories, sector_breakdown, buy_calls, sell_calls):
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    label = "Dzienny" if period == "daily" else "Tygodniowy"

    lines = [f"# {label} raport wplywu newsow na rynek — {today}", ""]
    lines.append(f"_{config.DISCLAIMER}_")
    lines.append("")
    lines.append(f"Przeanalizowano **{len(news_rows)}** newsow w tym okresie.")
    lines.append("")

    lines.append("## Najwazniejsze wydarzenia")
    if top_stories:
        for s in top_stories:
            lines.append(
                f"- **[{s['impact']:.1f}/10, {s['direction']}]** {s['title']} "
                f"({s['category']}) — {s['summary']}"
            )
    else:
        lines.append("- Brak newsow w tym okresie.")
    lines.append("")

    lines.append("## Rekomendacje KUPUJ")
    if buy_calls:
        for r in buy_calls:
            lines.append(
                f"- **{r['ticker']}** ({r['company']}) — score {r['score']:+.2f}, "
                f"pewnosc {r['confidence']:.2f}, {r['news_count']} newsow"
            )
    else:
        lines.append("- Brak sygnalow kupna w tym okresie.")
    lines.append("")

    lines.append("## Rekomendacje SPRZEDAJ")
    if sell_calls:
        for r in sell_calls:
            lines.append(
                f"- **{r['ticker']}** ({r['company']}) — score {r['score']:+.2f}, "
                f"pewnosc {r['confidence']:.2f}, {r['news_count']} newsow"
            )
    else:
        lines.append("- Brak sygnalow sprzedazy w tym okresie.")
    lines.append("")

    lines.append("## Wplyw wg sektora")
    if sector_breakdown:
        for s in sector_breakdown:
            lines.append(
                f"- **{s['sector']}**: {s['count']} newsow, "
                f"sredni impact {s['avg_impact']:.1f}/10, tickery: {', '.join(s['tickers'])}"
            )
    else:
        lines.append("- Brak dopasowanych spolek w tym okresie.")
    lines.append("")

    lines.append("## Wplyw wg kategorii RSS")
    for cat, e in sorted(by_category.items(), key=lambda kv: kv[1]["impact_sum"], reverse=True):
        avg = e["impact_sum"] / e["count"] if e["count"] else 0
        lines.append(f"- **{cat}**: {e['count']} newsow, sredni impact {avg:.1f}/10")

    return "\n".join(lines)


def _save_digest(period, md, top_movers):
    db = get_db()
    cur = db.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute(
        """
        INSERT INTO digests (date, period, summary, top_movers, created_at)
        VALUES (?,?,?,?,?)
        ON CONFLICT(date, period) DO UPDATE SET
            summary=excluded.summary,
            top_movers=excluded.top_movers,
            created_at=excluded.created_at
        """,
        (today, period, md, json.dumps(top_movers), datetime.now().isoformat()),
    )
    db.commit()


def _write_report_file(period, md):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(config.REPORTS_DIR, f"digest_{period}_{today}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[i] Zapisano raport: {filename}")
