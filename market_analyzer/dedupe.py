# ============================================================
# EVENT CLUSTERING / CORROBORATION
#
# Different outlets often cover the exact same event (an earnings
# beat, a recall) within hours of each other. Counting each as an
# independent bullish/bearish vote would let a single story get
# amplified just because five sites republished it - the same
# "echo chamber" bias documented in meme-stock/social-media research.
#
# Here we cluster same-ticker articles by title similarity within a
# rolling window. All articles in a cluster share one event_cluster id
# stored on the `assets` row. recommend.py then treats a cluster as
# ONE story (using its highest-impact article) but tracks how many
# independent sources corroborated it - which is itself a (capped)
# positive signal, distinct from raw repetition.
# ============================================================
import hashlib
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from .db import get_db

CLUSTER_WINDOW_HOURS = 48
SIMILARITY_THRESHOLD = 0.6


def _similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def assign_event_cluster(ticker, title, news_date=None):
    news_date = news_date or datetime.now()
    db = get_db()
    cur = db.cursor()
    cutoff = (news_date - timedelta(hours=CLUSTER_WINDOW_HOURS)).isoformat()

    candidates = cur.execute(
        """
        SELECT n.title, a.event_cluster
        FROM assets a
        JOIN news n ON n.id = a.news_id
        WHERE a.ticker = ? AND n.date >= ? AND a.event_cluster IS NOT NULL
        """,
        (ticker, cutoff),
    ).fetchall()

    for candidate_title, cluster_id in candidates:
        if _similar(title, candidate_title) >= SIMILARITY_THRESHOLD:
            return cluster_id

    seed = f"{ticker}:{title.lower().strip()}:{news_date.isoformat()}"
    return hashlib.md5(seed.encode()).hexdigest()[:16]


def corroboration_count(ticker, cluster_id):
    if not cluster_id:
        return 1
    db = get_db()
    cur = db.cursor()
    row = cur.execute(
        "SELECT COUNT(DISTINCT news_id) FROM assets WHERE ticker=? AND event_cluster=?",
        (ticker, cluster_id),
    ).fetchone()
    return row[0] if row else 1


def corroboration_weight(count):
    """Diminishing-returns bonus for independent confirmation, capped so
    a pile of copycat articles can't dominate the signal the way raw
    repetition would."""
    import math

    return min(1.0 + 0.15 * math.log1p(max(count, 1) - 1), 1.4)
