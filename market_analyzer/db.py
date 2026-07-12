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
            category TEXT
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
            confidence REAL
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
            UNIQUE(news_id, symbol)
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
            generated_at TEXT
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

    db.commit()
