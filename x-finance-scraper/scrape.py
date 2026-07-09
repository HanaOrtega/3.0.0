"""
Pobiera posty z X (Twitter) na temat firmy/hasla powiazanego z finansami
z ostatnich N godzin (domyslnie 72h), dzialajac przez przegladarke
(Playwright + prawdziwy DOM x.com), a NIE przez oficjalne API.

Wymaga wczesniejszego jednorazowego logowania: `python login.py`
(zapisuje sesje do auth/storage_state.json).

Uzycie:
    python scrape.py
    python scrape.py --query "Google" --hours 72
    python scrape.py --query "Google" --keywords "finance,earnings,stock" --lang en
    python scrape.py --headful          (pokaz okno przegladarki, jesli jest ekran)
    python scrape.py --strict           (odrzucaj posty bez slowa kluczowego z finansow)
"""
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_STATE_PATH = os.path.join(ROOT, "auth", "storage_state.json")
OUTPUT_DIR = os.path.join(ROOT, "output")
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "config", "default.json")

# Wykonywane w kontekscie strony (JS) - odczytuje dane z kazdej karty posta.
EXTRACT_JS = r"""
(articles) => articles.map((article) => {
    const timeEl = article.querySelector('time');
    const datetime = timeEl ? timeEl.getAttribute('datetime') : null;
    const linkEl = timeEl ? timeEl.closest('a') : null;
    const url = linkEl ? linkEl.href : null;

    const textEl = article.querySelector('[data-testid="tweetText"]');
    const text = textEl ? textEl.innerText.replace(/\s+/g, ' ').trim() : '';

    let handle = null;
    let displayName = null;
    const userNameBlock = article.querySelector('[data-testid="User-Name"]');
    if (userNameBlock) {
        const links = userNameBlock.querySelectorAll('a[href^="/"]');
        for (const a of links) {
            const href = a.getAttribute('href') || '';
            if (/^\/[A-Za-z0-9_]+$/.test(href)) {
                handle = href.slice(1);
                break;
            }
        }
        displayName = (userNameBlock.innerText.split('\n')[0] || '').trim() || null;
    }

    const metrics = {};
    ['reply', 'retweet', 'like'].forEach((key) => {
        const el = article.querySelector(`[data-testid="${key}"]`);
        metrics[key] = el ? (el.getAttribute('aria-label') || el.textContent || '').trim() : null;
    });

    return { url, datetime, text, handle, displayName, metrics };
})
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Scraper postow z X (Twitter) przez przegladarke.")
    parser.add_argument("--query", type=str, help="Glowne haslo/firma do wyszukania")
    parser.add_argument("--keywords", type=str, help="Slowa kluczowe oddzielone przecinkami (OR)")
    parser.add_argument("--hours", type=float, help="Ile godzin wstecz pobierac posty")
    parser.add_argument("--lang", type=str, help="Kod jezyka X (np. pl, en)")
    parser.add_argument("--max-tweets", type=int, help="Limit liczby zebranych postow")
    parser.add_argument("--max-scrolls", type=int, help="Limit liczby scrolli strony wynikow")
    parser.add_argument("--strict", action="store_true", help="Post musi zawierac haslo ORAZ slowo finansowe")
    parser.add_argument("--no-exclude-replies", action="store_true", help="Nie wykluczaj odpowiedzi (replies)")
    parser.add_argument("--headful", action="store_true", help="Pokaz okno przegladarki")
    return parser.parse_args()


def load_config(args):
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.query:
        cfg["query"] = args.query
    if args.keywords:
        cfg["keywords"] = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if args.hours is not None:
        cfg["hoursBack"] = args.hours
    if args.lang:
        cfg["lang"] = args.lang
    if args.max_tweets:
        cfg["maxTweets"] = args.max_tweets
    if args.max_scrolls:
        cfg["maxScrolls"] = args.max_scrolls
    if args.strict:
        cfg["strictKeywordFilter"] = True
    if args.no_exclude_replies:
        cfg["excludeReplies"] = False

    cfg["headless"] = not args.headful
    return cfg


def build_search_query(cfg):
    parts = [cfg["query"]]
    keywords = cfg.get("keywords") or []
    if keywords:
        or_clause = " OR ".join(f'"{k}"' if " " in k else k for k in keywords)
        parts.append(f"({or_clause})")
    if cfg.get("excludeReplies"):
        parts.append("-filter:replies")
    if cfg.get("lang"):
        parts.append(f"lang:{cfg['lang']}")
    return " ".join(parts)


def matches_keywords(text, cfg):
    lower = text.lower()
    if not cfg.get("strictKeywordFilter"):
        return True
    query_hit = cfg["query"].lower() in lower
    kw_hit = any(k.lower() in lower for k in cfg.get("keywords", []))
    return query_hit and kw_hit


def parse_dt(iso_str):
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))


def write_csv(path, rows):
    headers = ["datetime", "handle", "displayName", "text", "url", "reply", "retweet", "like"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            metrics = r.get("metrics") or {}
            writer.writerow([
                r.get("datetime"),
                r.get("handle"),
                r.get("displayName"),
                r.get("text"),
                r.get("url"),
                metrics.get("reply"),
                metrics.get("retweet"),
                metrics.get("like"),
            ])


def main():
    args = parse_args()
    cfg = load_config(args)

    if not os.path.exists(STORAGE_STATE_PATH):
        print(
            f"Brak pliku sesji: {STORAGE_STATE_PATH}\n"
            "Najpierw zaloguj sie jednorazowo lokalnie: python login.py",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg["hoursBack"])
    search_query = build_search_query(cfg)
    url = f"https://x.com/search?q={quote(search_query)}&src=typed_query&f=live"

    print(f"Zapytanie wyszukiwania: {search_query}")
    print(f"Okno czasowe: ostatnie {cfg['hoursBack']}h (od {cutoff.isoformat()})")

    seen = {}
    stale_scrolls = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=cfg["headless"])
        context = browser.new_context(
            storage_state=STORAGE_STATE_PATH,
            viewport={"width": 1280, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")

        if "/login" in page.url or "/i/flow/login" in page.url:
            print("Sesja wygasla lub jest nieprawidlowa - zaloguj sie ponownie: python login.py", file=sys.stderr)
            browser.close()
            sys.exit(1)

        try:
            page.wait_for_selector('article[data-testid="tweet"]', timeout=20000)
        except Exception:
            print("Nie znaleziono zadnych postow dla tego zapytania (lub strona nie zaladowala sie poprawnie).")

        for i in range(cfg["maxScrolls"]):
            batch = page.eval_on_selector_all('article[data-testid="tweet"]', EXTRACT_JS)

            for t in batch:
                if not t.get("url") or not t.get("datetime"):
                    continue
                if t["url"] in seen:
                    continue
                if not matches_keywords(t.get("text", ""), cfg):
                    continue
                seen[t["url"]] = t

            times = [parse_dt(t["datetime"]) for t in batch if t.get("datetime")]
            oldest_in_batch = min(times) if times else None

            if oldest_in_batch is not None and oldest_in_batch < cutoff:
                stale_scrolls += 1
            else:
                stale_scrolls = 0

            print(f"Scroll {i + 1}/{cfg['maxScrolls']} - zebrano lacznie: {len(seen)}")

            if stale_scrolls >= 3:
                print("Kolejne posty sa juz starsze niz zadane okno czasowe - konczenie.")
                break
            if len(seen) >= cfg["maxTweets"]:
                print("Osiagnieto limit maxTweets - konczenie.")
                break

            page.mouse.wheel(0, 2200)
            page.wait_for_timeout(1200 + random.random() * 1300)

        browser.close()

    results = [t for t in seen.values() if parse_dt(t["datetime"]) >= cutoff]
    results.sort(key=lambda t: parse_dt(t["datetime"]), reverse=True)

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    json_path = os.path.join(OUTPUT_DIR, f"x-posts-{stamp}.json")
    csv_path = os.path.join(OUTPUT_DIR, f"x-posts-{stamp}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    write_csv(csv_path, results)

    print(f"\nZnaleziono {len(results)} postow z ostatnich {cfg['hoursBack']}h.")
    print(f"Zapisano do:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
