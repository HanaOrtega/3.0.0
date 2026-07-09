"""
Pobieranie surowych wpisow z roznych zrodel newsowych:
- statyczne kanaly RSS (dowolna lista)
- Google News RSS (wyszukiwanie po hasle, bez klucza API)
- NewsAPI (https://newsapi.org, wymaga NEWSAPI_KEY)
- GNews (https://gnews.io, wymaga GNEWS_API_KEY)
"""
import os
import time
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _parsed_time_to_iso(struct_time):
    if not struct_time:
        return None
    return datetime.fromtimestamp(time.mktime(struct_time), tz=timezone.utc).isoformat()


def fetch_rss_feed(url, source_name=None, limit=25):
    articles = []
    try:
        feed = feedparser.parse(url, agent=USER_AGENT)
    except Exception as e:
        print(f"  [RSS] Blad pobierania {url}: {e}")
        return articles

    feed_title = source_name or (feed.feed.get("title") if getattr(feed, "feed", None) else None) or url

    for entry in feed.entries[:limit]:
        published = _parsed_time_to_iso(getattr(entry, "published_parsed", None))
        articles.append({
            "title": (entry.get("title") or "").strip(),
            "url": (entry.get("link") or "").strip(),
            "published": published,
            "description": (entry.get("summary", "") or "").strip(),
            "source": feed_title,
            "origin": "rss",
        })
    return articles


def fetch_google_news_rss(query, lang="pl", country="PL", limit=25):
    q = quote(query)
    ceid = f"{country}:{lang}"
    url = f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={ceid}"
    return fetch_rss_feed(url, source_name="Google News", limit=limit)


def fetch_static_rss_feeds(feeds, limit=25):
    articles = []
    for url in feeds:
        articles.extend(fetch_rss_feed(url, limit=limit))
    return articles


def fetch_newsapi(query, lang="pl", limit=25, api_key=None):
    api_key = api_key or os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": lang,
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [NewsAPI] Blad zapytania: {e}")
        return []

    articles = []
    for item in data.get("articles", []):
        articles.append({
            "title": (item.get("title") or "").strip(),
            "url": (item.get("url") or "").strip(),
            "published": item.get("publishedAt"),
            "description": (item.get("description") or "").strip(),
            "source": (item.get("source") or {}).get("name") or "NewsAPI",
            "origin": "newsapi",
        })
    return articles


def fetch_gnews(query, lang="pl", country="pl", limit=25, api_key=None):
    api_key = api_key or os.environ.get("GNEWS_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": query,
                "lang": lang,
                "country": country,
                "max": min(limit, 100),
                "apikey": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [GNews] Blad zapytania: {e}")
        return []

    articles = []
    for item in data.get("articles", []):
        articles.append({
            "title": (item.get("title") or "").strip(),
            "url": (item.get("url") or "").strip(),
            "published": item.get("publishedAt"),
            "description": (item.get("description") or "").strip(),
            "source": (item.get("source") or {}).get("name") or "GNews",
            "origin": "gnews",
        })
    return articles
