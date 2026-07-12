# ============================================================
# CALIBRATION - the "learn from mistakes" layer
#
# Every resolved backtest (see backtest.py) tells us whether the AI's
# call was right or wrong for a given news SOURCE, EVENT TYPE and
# SECTOR. We fold that outcome into a Beta-Bernoulli posterior per
# dimension/key, so the system keeps a running, ever-updating
# "how much should I trust this kind of signal" score - the same
# idea used for news-source-credibility weighting in the literature,
# and for the persistent reflection/memory step in projects like
# TauricResearch/TradingAgents (which re-injects outcome-grounded
# lessons into future decisions rather than treating every article
# as a first-time judgement).
#
# Cold start: instead of a flat 50/50 prior for every source, we seed
# mainstream wire services slightly above neutral and low-moderation
# social feeds slightly below - a mild informed prior grounded in the
# news-credibility research, NOT a claim of measured accuracy. Once
# enough backtests accumulate, the data overrides the prior.
# ============================================================
from datetime import datetime

from .db import get_db

_WIRE_SOURCES = {
    "associated press business", "reuters markets", "bloomberg markets",
    "financial times markets", "bbc business", "cnbc markets",
    "marketwatch top stories", "the economist — finance", "the guardian business",
}
_SOCIAL_SOURCES = {
    "r/wallstreetbets", "r/stocks", "r/investing", "r/options", "r/valueinvesting",
}


def _default_prior(dimension, key):
    if dimension == "source":
        lowered = (key or "").lower()
        if lowered in _WIRE_SOURCES:
            return 6.0, 4.0  # mild positive prior, ~0.60
        if lowered in _SOCIAL_SOURCES:
            return 4.0, 6.0  # mild negative prior, ~0.40
    return 5.0, 5.0  # neutral, ~0.50


def _get_row(cur, dimension, key):
    row = cur.execute(
        "SELECT alpha, beta, n FROM signal_calibration WHERE dimension=? AND key=?",
        (dimension, key),
    ).fetchone()
    if row:
        return row
    prior_a, prior_b = _default_prior(dimension, key)
    cur.execute(
        "INSERT INTO signal_calibration (dimension, key, alpha, beta, n, last_updated) VALUES (?,?,?,?,0,?)",
        (dimension, key, prior_a, prior_b, datetime.now().isoformat()),
    )
    return prior_a, prior_b, 0


def update_from_backtest(source=None, event_type=None, sector=None, correct=None):
    if correct is None:
        return
    db = get_db()
    cur = db.cursor()
    for dimension, key in (("source", source), ("event_type", event_type), ("sector", sector)):
        if not key:
            continue
        alpha, beta, n = _get_row(cur, dimension, key)
        if correct:
            alpha += 1
        else:
            beta += 1
        cur.execute(
            """
            INSERT INTO signal_calibration (dimension, key, alpha, beta, n, last_updated)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(dimension, key) DO UPDATE SET
                alpha=excluded.alpha, beta=excluded.beta, n=excluded.n, last_updated=excluded.last_updated
            """,
            (dimension, key, alpha, beta, n + 1, datetime.now().isoformat()),
        )
    db.commit()


def _posterior_mean(dimension, key):
    if not key:
        return None, 0
    db = get_db()
    cur = db.cursor()
    row = cur.execute(
        "SELECT alpha, beta, n FROM signal_calibration WHERE dimension=? AND key=?",
        (dimension, key),
    ).fetchone()
    if not row:
        prior_a, prior_b = _default_prior(dimension, key)
        return prior_a / (prior_a + prior_b), 0
    alpha, beta, n = row
    return alpha / (alpha + beta), n


def get_weight(source=None, event_type=None, sector=None):
    """Combine trust in source/event_type/sector into a single multiplier
    centered on 1.0 (0.5 = fully distrust that dimension's contribution,
    1.5 = fully trust it). Missing dimensions simply don't contribute."""
    multipliers = []
    for dimension, key in (("source", source), ("event_type", event_type), ("sector", sector)):
        mean, _ = _posterior_mean(dimension, key)
        if mean is None:
            continue
        multipliers.append(0.5 + mean)  # mean in [0,1] -> multiplier in [0.5, 1.5]

    if not multipliers:
        return 1.0

    product = 1.0
    for m in multipliers:
        product *= m
    return product ** (1.0 / len(multipliers))  # geometric mean


def snapshot(dimension):
    db = get_db()
    cur = db.cursor()
    rows = cur.execute(
        "SELECT key, alpha, beta, n FROM signal_calibration WHERE dimension=? ORDER BY n DESC",
        (dimension,),
    ).fetchall()
    return [
        {"key": key, "accuracy": alpha / (alpha + beta), "n": n}
        for key, alpha, beta, n in rows
    ]
