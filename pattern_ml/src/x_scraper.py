"""Wyszukiwanie świeżych wzmianek o instrumencie na X (Twitter) - tylko zakładka
'Najnowsze' (f=live), żeby wykluczyć stare/archiwalne posty przebite przez
algorytm popularności."""

import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from playwright.sync_api import Page

SEARCH_URL = "https://x.com/search?q={query}&src=typed_query&f=live"
TWEET_SELECTOR = 'article[data-testid="tweet"]'

_COUNT_RE = re.compile(r"^([\d.,]+)\s*([KMB]?)", re.IGNORECASE)
_MULTIPLIERS = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def _parse_count(aria_label: str | None) -> int:
    if not aria_label:
        return 0
    match = _COUNT_RE.match(aria_label.strip())
    if not match:
        return 0
    number = float(match.group(1).replace(",", ""))
    return int(number * _MULTIPLIERS[match.group(2).upper()])


def _extract_tweet(article) -> dict | None:
    time_el = article.locator("time").first
    if time_el.count() == 0:
        return None
    published_at = time_el.get_attribute("datetime")

    link = article.locator('a:has(time)').first
    href = link.get_attribute("href") if link.count() else None
    url = f"https://x.com{href}" if href else None

    text_el = article.locator('[data-testid="tweetText"]')
    text = text_el.first.inner_text() if text_el.count() else ""

    name_el = article.locator('[data-testid="User-Name"]')
    author_raw = name_el.first.inner_text() if name_el.count() else ""
    handle_match = re.search(r"@\w+", author_raw)
    author_handle = handle_match.group(0) if handle_match else ""
    author_name = author_raw[: handle_match.start()].strip() if handle_match else author_raw.strip()

    def count_for(testid: str) -> int:
        el = article.locator(f'[data-testid="{testid}"]')
        if el.count() == 0:
            return 0
        return _parse_count(el.first.get_attribute("aria-label"))

    return {
        "url": url,
        "author_name": author_name,
        "author_handle": author_handle,
        "text": text,
        "published_at": published_at,
        "replies": count_for("reply"),
        "retweets": count_for("retweet"),
        "likes": count_for("like"),
    }


def _scroll_and_collect(page: Page, max_scrolls: int, min_wait: float = 1.5) -> list[dict]:
    seen_urls = set()
    results = []

    try:
        page.wait_for_selector(TWEET_SELECTOR, timeout=15000)
    except Exception:
        return results  # brak wyników dla tego zapytania

    for _ in range(max_scrolls):
        articles = page.locator(TWEET_SELECTOR)
        count = articles.count()
        for i in range(count):
            item = _extract_tweet(articles.nth(i))
            if item and item["url"] and item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                results.append(item)

        page.mouse.wheel(0, 2500)
        time.sleep(min_wait)

    return results


def search_query(page: Page, query: str, max_scrolls: int = 5) -> list[dict]:
    """Wykonuje wyszukiwanie na X (zakładka Najnowsze) i zwraca listę postów."""
    page.goto(SEARCH_URL.format(query=quote(query)), wait_until="domcontentloaded")
    time.sleep(2)
    return _scroll_and_collect(page, max_scrolls=max_scrolls)


def fetch_fresh_mentions(
    page: Page,
    ticker: str,
    company: str | None,
    accounts: list[str],
    hours: int,
    max_scrolls: int = 5,
) -> list[dict]:
    """Zbiera świeże (max `hours` godzin) wzmianki o instrumencie: ogólne wyszukiwanie
    po cashtagu/nazwie spółki + wyszukiwanie w obrębie wskazanych kont."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    queries = [f"${ticker}"]
    if company:
        queries.append(f'"{company}"')
    for account in accounts:
        handle = account.lstrip("@")
        terms = f"${ticker}" + (f' OR "{company}"' if company else "")
        queries.append(f"(from:{handle}) ({terms})")

    all_items = {}
    for query in queries:
        for item in search_query(page, query, max_scrolls=max_scrolls):
            if item["url"] in all_items:
                continue
            if not item["published_at"]:
                continue
            published = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
            if published < cutoff:
                continue
            item["published_at"] = published.isoformat()
            item["matched_query"] = query
            all_items[item["url"]] = item

    return sorted(all_items.values(), key=lambda x: x["published_at"], reverse=True)
