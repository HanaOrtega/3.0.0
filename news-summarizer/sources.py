"""
Pobieranie surowych wpisow z roznych zrodel newsowych:
- statyczne kanaly RSS (dowolna lista)
- Google News RSS (wyszukiwanie po hasle, bez klucza API)
- NewsAPI (https://newsapi.org, wymaga NEWSAPI_KEY)
- GNews (https://gnews.io, wymaga GNEWS_API_KEY)
- strony z listingiem newsow danej spolki (np. investing.com/equities/...) -
  generyczny scraping HTML, bez RSS/API
"""
import os
import time
from datetime import datetime, timezone
from urllib.parse import quote, urljoin

import feedparser
import requests
from bs4 import BeautifulSoup

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


def _parse_investing_datetime(value):
    """Format obserwowany na investing.com: '2026-07-09 19:49:58' (bez strefy - traktujemy jako UTC)."""
    if not value:
        return None
    try:
        dt = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def _extract_investing_articles(soup, base_url, source_name):
    """
    Selektory dopasowane do investing.com (strony 'News' danej spolki, np.
    investing.com/equities/<spolka>-news) - kazdy wpis to
    article[data-test="article-item"] z linkiem, opisem, data i zrodlem.
    """
    articles = []
    for art in soup.select('article[data-test="article-item"]'):
        title_link = art.select_one('a[data-test="article-title-link"]')
        if not title_link:
            continue
        href = title_link.get("href") or ""
        title = title_link.get_text(strip=True)
        if not href or not title:
            continue

        desc_el = art.select_one('p[data-test="article-description"]')
        description = desc_el.get_text(strip=True) if desc_el else ""

        time_el = art.select_one('time[data-test="article-publish-date"]')
        published = _parse_investing_datetime(time_el.get("datetime")) if time_el else None

        provider_el = art.select_one('a[data-test="article-provider-link"]')
        provider = provider_el.get_text(strip=True) if provider_el else (source_name or "Investing.com")

        articles.append({
            "title": title,
            "url": urljoin(base_url, href),
            "published": published,
            "description": description,
            "source": provider,
            "origin": "scrape",
        })
    return articles


def _extract_generic_articles(soup, base_url, source_name):
    """Fallback dla stron o innej strukturze niz investing.com - defensywne selektory."""
    articles = []
    name = source_name or base_url
    seen = set()

    for selector in ("a.title[href]", "article a[href]", 'a[href*="/news/"]'):
        links = soup.select(selector)
        if not links:
            continue
        for link in links:
            href = link.get("href") or ""
            text = link.get_text(strip=True)
            if not href or not text or len(text) < 15:
                continue
            full_url = urljoin(base_url, href)
            if full_url in seen:
                continue
            seen.add(full_url)
            articles.append({
                "title": text,
                "url": full_url,
                "published": None,
                "description": "",
                "source": name,
                "origin": "scrape",
            })
        if articles:
            break
    return articles


def fetch_listing_page(url, source_name=None, limit=25, max_pages=1):
    """
    Scraper strony z lista newsow danej spolki (np. investing.com/equities/
    <spolka>-news) - bez RSS/API. Probuje najpierw selektorow dopasowanych do
    investing.com, a jesli nie znajdzie nic pasujacego (inny serwis), spada
    do generycznej heurystyki. `max_pages` > 1 pobiera kolejne strony
    (investing.com uzywa wzorca .../<spolka>-news/2, /3, ...).
    """
    articles = []
    seen_urls = set()

    for page_num in range(1, max_pages + 1):
        page_url = url if page_num == 1 else f"{url.rstrip('/')}/{page_num}"
        try:
            resp = requests.get(
                page_url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "pl,en;q=0.8"},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"  [Listing] Blad pobierania {page_url}: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        page_articles = _extract_investing_articles(soup, page_url, source_name)
        if not page_articles:
            page_articles = _extract_generic_articles(soup, page_url, source_name)

        new_count = 0
        for a in page_articles:
            if a["url"] in seen_urls:
                continue
            seen_urls.add(a["url"])
            articles.append(a)
            new_count += 1

        if new_count == 0 or len(articles) >= limit:
            break

    return articles[:limit]


def fetch_listing_pages(pages, limit=25):
    """pages: lista {"url": ..., "source": ..., "maxPages": ...} (lub samych URL-i)."""
    articles = []
    for page in pages:
        if isinstance(page, dict):
            url = page.get("url")
            source_name = page.get("source")
            max_pages = page.get("maxPages", 1)
        else:
            url = page
            source_name = None
            max_pages = 1
        if not url:
            continue
        articles.extend(fetch_listing_page(url, source_name=source_name, limit=limit, max_pages=max_pages))
    return articles
