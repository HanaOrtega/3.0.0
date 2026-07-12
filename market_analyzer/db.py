# ============================================================
# DATABASE
# ============================================================
import sqlite3

from . import config

_connection = None


def get_db():
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(config.DB_FILE, check_same_thread=False)
    return _connection


def init_database():
    db = get_db()
    cur = db.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY,
            hash TEXT UNIQUE,
            date TEXT,
            title TEXT,
            url TEXT,
            content TEXT,
            analysis TEXT,
            category TEXT,
            feed_group TEXT,
            event_type TEXT,
            time_horizon_hours REAL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY,
            news_id INTEGER,
            company TEXT,
            ticker TEXT,
            exchange TEXT,
            sector TEXT,
            confidence REAL,
            event_cluster TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS impact_backtest (
            id INTEGER PRIMARY KEY,
            news_id INTEGER,
            symbol TEXT,
            predicted_direction TEXT,
            impact_score REAL,
            price_at_news REAL,
            price_after_4h REAL,
            price_after_24h REAL,
            return_4h REAL,
            return_24h REAL,
            was_correct INTEGER,
            checked_at TEXT,
            benchmark_symbol TEXT,
            benchmark_return_4h REAL,
            benchmark_return_24h REAL,
            alpha_4h REAL,
            alpha_24h REAL,
            barrier_hit TEXT,
            label INTEGER,
            UNIQUE(news_id, symbol)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_calibration (
            id INTEGER PRIMARY KEY,
            dimension TEXT,
            key TEXT,
            alpha REAL DEFAULT 1.0,
            beta REAL DEFAULT 1.0,
            n INTEGER DEFAULT 0,
            last_updated TEXT,
            UNIQUE(dimension, key)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY,
            topic TEXT,
            summary TEXT,
            importance REAL,
            news_ids TEXT,
            assets TEXT,
            created TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            company TEXT,
            sector TEXT,
            action TEXT,
            score REAL,
            confidence REAL,
            avg_sentiment REAL,
            news_count INTEGER,
            backtest_accuracy REAL,
            rationale TEXT,
            generated_at TEXT,
            score_today REAL,
            news_count_today INTEGER,
            high_conviction_today INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS digests (
            id INTEGER PRIMARY KEY,
            date TEXT,
            period TEXT,
            summary TEXT,
            top_movers TEXT,
            created_at TEXT,
            UNIQUE(date, period)
        )
        """
    )

    _migrate_existing_tables(db)
    db.commit()


# Idempotent ALTER TABLE for databases created before these columns existed.
_MIGRATIONS = {
    "news": ["feed_group TEXT", "event_type TEXT", "time_horizon_hours REAL"],
    "assets": ["event_cluster TEXT"],
    "impact_backtest": [
        "benchmark_symbol TEXT",
        "benchmark_return_4h REAL",
        "benchmark_return_24h REAL",
        "alpha_4h REAL",
        "alpha_24h REAL",
        "barrier_hit TEXT",
        "label INTEGER",
    ],
    "recommendations": [
        "score_today REAL",
        "news_count_today INTEGER",
        "high_conviction_today INTEGER",
    ],
}


def _migrate_existing_tables(db):
    cur = db.cursor()
    for table, columns in _MIGRATIONS.items():
        for column_def in columns:
            column_name = column_def.split()[0]
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
