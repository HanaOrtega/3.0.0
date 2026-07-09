"""
Pobiera newsy z wielu zrodel (statyczne kanaly RSS, Google News RSS, NewsAPI,
GNews, strony z listingiem newsow per-spolka np. investing.com) wraz z pelna
trescia artykulow (generyczny scraping stron przez trafilatura) i podsumowuje
je lokalnym LLM (Ollama) - bez wysylania danych do zadnego platnego API do
podsumowan.

Wymaga uruchomionej Ollama (`ollama serve`) z pobranym modelem, np.:
    ollama pull llama3

Tematy do przetworzenia bierze domyslnie z `config/default.json` (klucz
"queries" - lista, np. ["Google", "Apple"]) - dla kazdego tematu z listy
generowany jest osobny raport. Flaga --query nadpisuje liste i przetwarza
tylko jeden, podany temat.

Uzycie:
    python main.py                              # wszystkie tematy z config/default.json
    python main.py --query "Google" --hours 72   # tylko jeden, wskazany temat
    python main.py --query "Tesla" --max-articles 20 --no-fulltext
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from dateutil import parser as dateparser

import extract
import report
import sources
import summarize

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "config", "default.json")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pobieranie i podsumowywanie newsow z wielu zrodel przy uzyciu lokalnego LLM."
    )
    p.add_argument("--query", type=str, help="Temat/haslo wyszukiwania (nadpisuje liste 'queries' z configu)")
    p.add_argument("--hours", type=float, help="Ile godzin wstecz brac artykuly")
    p.add_argument("--lang", type=str, help="Kod jezyka (np. pl, en)")
    p.add_argument("--country", type=str, help="Kod kraju (np. PL, US) - dla Google News/GNews")
    p.add_argument("--model", type=str, help="Nazwa modelu Ollama")
    p.add_argument("--ollama-host", type=str, help="Adres serwera Ollama")
    p.add_argument("--max-articles", type=int, help="Maksymalna liczba artykulow do podsumowania")
    p.add_argument("--no-fulltext", action="store_true", help="Nie pobieraj pelnej tresci (uzyj opisu z RSS/API)")
    p.add_argument("--no-rss", action="store_true", help="Pomin statyczne kanaly RSS")
    p.add_argument("--no-google-news", action="store_true", help="Pomin Google News RSS")
    p.add_argument("--no-newsapi", action="store_true", help="Pomin NewsAPI")
    p.add_argument("--no-gnews", action="store_true", help="Pomin GNews")
    p.add_argument("--no-listing-pages", action="store_true", help="Pomin strony z listingiem newsow (np. investing.com)")
    return p.parse_args()


def load_config(args):
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.hours is not None:
        cfg["hoursBack"] = args.hours
    if args.lang:
        cfg["lang"] = args.lang
    if args.country:
        cfg["country"] = args.country
    if args.model:
        cfg["ollamaModel"] = args.model
    if args.ollama_host:
        cfg["ollamaHost"] = args.ollama_host
    if args.max_articles:
        cfg["maxTotalArticles"] = args.max_articles
    return cfg


def resolve_queries(cfg, args):
    if args.query:
        return [args.query]
    queries = cfg.get("queries")
    if queries:
        return queries
    if cfg.get("query"):
        return [cfg["query"]]
    return []


def normalize_url(url):
    return (url or "").split("?")[0].rstrip("/")


def parse_published(value):
    if not value:
        return None
    try:
        dt = dateparser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, OverflowError):
        return None


def collect_articles(query, cfg, args):
    all_articles = []

    if not args.no_rss:
        print("Pobieranie statycznych kanalow RSS...")
        all_articles.extend(sources.fetch_static_rss_feeds(cfg["rssFeeds"], limit=cfg["maxPerSource"]))

    if not args.no_google_news and cfg.get("useGoogleNewsRss", True):
        print("Pobieranie Google News RSS...")
        all_articles.extend(sources.fetch_google_news_rss(
            query, lang=cfg["lang"], country=cfg["country"], limit=cfg["maxPerSource"]
        ))

    if not args.no_newsapi:
        print("Pobieranie z NewsAPI (jesli ustawiony NEWSAPI_KEY)...")
        all_articles.extend(sources.fetch_newsapi(query, lang=cfg["lang"], limit=cfg["maxPerSource"]))

    if not args.no_gnews:
        print("Pobieranie z GNews (jesli ustawiony GNEWS_API_KEY)...")
        all_articles.extend(sources.fetch_gnews(
            query, lang=cfg["lang"], country=cfg["country"].lower(), limit=cfg["maxPerSource"]
        ))

    if not args.no_listing_pages:
        pages = (cfg.get("listingPages") or {}).get(query, [])
        if pages:
            print(f"Pobieranie stron z listingiem newsow ({len(pages)}) dla '{query}'...")
            all_articles.extend(sources.fetch_listing_pages(pages, limit=cfg["maxPerSource"]))

    return all_articles


def dedupe_and_filter(articles, cutoff):
    seen_urls = set()
    seen_titles = set()
    result = []
    for a in articles:
        url_key = normalize_url(a.get("url"))
        title_key = (a.get("title") or "").strip().lower()
        if not url_key or not title_key:
            continue
        if url_key in seen_urls or title_key in seen_titles:
            continue

        published_dt = parse_published(a.get("published"))
        if published_dt is not None and published_dt < cutoff:
            continue

        seen_urls.add(url_key)
        seen_titles.add(title_key)
        result.append(a)
    return result


def process_query(query, cfg, args):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg["hoursBack"])
    print(f"\n=== Temat: {query} | okno czasowe: ostatnie {cfg['hoursBack']}h ===")

    raw_articles = collect_articles(query, cfg, args)
    print(f"Pobrano lacznie {len(raw_articles)} wpisow (przed deduplikacja i filtrem czasu).")

    articles = dedupe_and_filter(raw_articles, cutoff)
    articles = articles[: cfg["maxTotalArticles"]]
    print(f"Po deduplikacji i filtrze czasu: {len(articles)} artykulow.")

    if not articles:
        print("Brak artykulow spelniajacych kryteria - pomijam ten temat.")
        return

    host = cfg["ollamaHost"]
    model = cfg["ollamaModel"]
    llm_available = summarize.check_ollama_available(host)
    if not llm_available:
        print(
            f"UWAGA: nie mozna polaczyc sie z Ollama pod {host}.\n"
            f"Uruchom 'ollama serve' i sprawdz, czy model jest pobrany (ollama pull {model}).\n"
            "Raport zostanie wygenerowany bez podsumowan LLM (tylko opisy ze zrodel)."
        )

    final_articles = []
    for i, a in enumerate(articles, 1):
        print(f"[{i}/{len(articles)}] {a['title'][:80]}")
        full_text = None
        if not args.no_fulltext:
            extracted = extract.extract_article(a["url"])
            full_text = extracted["text"]
            # Niektore zrodla (np. strony-listingi typu investing.com) nie
            # podaja daty przy linku - doszacowujemy ja teraz z metadanych
            # pelnego artykulu i dopiero teraz mozemy odrzucic, jesli jednak
            # jest spoza zadanego okna czasowego.
            if not a.get("published") and extracted["published"]:
                a["published"] = extracted["published"]
                pub_dt = parse_published(a["published"])
                if pub_dt is not None and pub_dt < cutoff:
                    print("  Pomijam - po pobraniu tresci data okazala sie spoza okna czasowego.")
                    continue
        text_for_summary = full_text or a.get("description") or a["title"]

        if llm_available:
            try:
                a["summary"] = summarize.summarize_article(
                    host, model, a["title"], text_for_summary, query, lang=cfg["lang"]
                )
            except summarize.OllamaError as e:
                print(f"  Blad LLM: {e}")
                a["summary"] = a.get("description") or None
        else:
            a["summary"] = a.get("description") or None

        final_articles.append(a)

    articles = final_articles
    if not articles:
        print("Brak artykulow spelniajacych kryteria po weryfikacji daty - pomijam ten temat.")
        return

    overall_summary = None
    if llm_available:
        summarized = [a for a in articles if a.get("summary")]
        if summarized:
            try:
                overall_summary = summarize.summarize_digest(
                    host, model, summarized, query, lang=cfg["lang"]
                )
            except summarize.OllamaError as e:
                print(f"Blad generowania podsumowania zbiorczego: {e}")

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    safe_query = "".join(c if c.isalnum() else "-" for c in query).strip("-").lower() or "query"
    md_path = os.path.join(OUTPUT_DIR, f"news-{safe_query}-{stamp}.md")
    json_path = os.path.join(OUTPUT_DIR, f"news-{safe_query}-{stamp}.json")

    report.write_markdown_report(md_path, query, cfg["hoursBack"], overall_summary, articles)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"query": query, "overall_summary": overall_summary, "articles": articles},
            f, ensure_ascii=False, indent=2,
        )

    print(f"Zapisano raport do:\n  {md_path}\n  {json_path}")


def main():
    args = parse_args()
    cfg = load_config(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    queries = resolve_queries(cfg, args)
    if not queries:
        print(
            'Brak zdefiniowanych tematow - dodaj "queries": ["Google", "Apple"] '
            "do config/default.json albo uzyj --query."
        )
        return

    for query in queries:
        process_query(query, cfg, args)


if __name__ == "__main__":
    main()
