# ============================================================
# NEWS INGESTION PIPELINE
# RSS -> download -> extract -> AI analysis -> store
# ============================================================
import hashlib
from datetime import datetime

import feedparser
from playwright.sync_api import sync_playwright

from . import companies, config, extract, fetch, llm
from .db import get_db


def save_news(title, url, content, analysis, category):
    db = get_db()
    cur = db.cursor()
    url_hash = hashlib.md5(url.encode()).hexdigest()
    try:
        cur.execute(
            """
            INSERT INTO news (hash, date, title, url, content, analysis, category)
            VALUES (?,?,?,?,?,?,?)
            """,
            (url_hash, datetime.now().isoformat(), title, url, content, analysis, category),
        )
        db.commit()
        return cur.lastrowid
    except Exception as e:
        if "UNIQUE" in str(e):
            return None
        raise


def save_assets(news_id, assets):
    db = get_db()
    cur = db.cursor()
    for asset in assets:
        cur.execute(
            """
            INSERT INTO assets (news_id, company, ticker, exchange, sector, confidence)
            VALUES (?,?,?,?,?,?)
            """,
            (
                news_id,
                asset["company"],
                asset["ticker"],
                asset["exchange"],
                asset.get("sector", ""),
                asset["confidence"],
            ),
        )
    db.commit()


def news_exists(url_hash):
    cur = get_db().cursor()
    return cur.execute("SELECT id FROM news WHERE hash=?", (url_hash,)).fetchone() is not None


def run_pipeline():
    print("\n==============================")
    print(" START AI MARKET ANALYZER ")
    print(datetime.now())
    print("==============================\n")

    company_map = companies.load_company_map()
    if not company_map:
        print("[!] Brak company_map.json - kontynuuje bez mapowania spolek")

    from . import feeds as feeds_module

    feed_list = feeds_module.load_opml(config.OPML_FILE)
    if not feed_list:
        print("[!] Brak RSS")
        return

    new_articles = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 Windows Chrome")

        for feed_info in feed_list:
            rss_url = feed_info["url"]
            category = feed_info["name"]

            print("\nRSS:", category)

            try:
                feed = feedparser.parse(rss_url)
            except Exception as e:
                print("[!] Feed parse error:", category, e)
                continue

            for entry in feed.entries[: config.MAX_RSS_ITEMS]:
                url = getattr(entry, "link", None)
                title = getattr(entry, "title", "")
                if not url:
                    continue

                article_hash = hashlib.md5(url.encode()).hexdigest()
                if news_exists(article_hash):
                    continue

                print("[+] Nowy:", title)

                html_file = fetch.download_article(context, url)
                text = extract.extract_article_text(html_file)

                if len(text) < config.MIN_ARTICLE_CHARS:
                    print(" - pominieto (za krotki tekst)")
                    continue

                analysis = llm.analyze_article(text)

                news_id = save_news(title, url, text, analysis, category)
                if not news_id:
                    continue

                new_articles += 1

                assets = companies.detect_assets(text, company_map)
                if assets:
                    save_assets(news_id, assets)
                    print("    -> aktywa:", ", ".join(a["ticker"] for a in assets))

        browser.close()

    print(f"\nKONIEC PIPELINE - nowych artykulow: {new_articles}")
