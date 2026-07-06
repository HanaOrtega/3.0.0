"""
Zbiera świeże (niedawne, nie archiwalne) wzmianki o instrumencie na X (Twitter)
i zapisuje je do pliku JSON - jako uzupełniający sygnał do analizy technicznej/ML
z main.py.

Wymaga zmiennych środowiskowych:
    X_USERNAME - login/e-mail/telefon do X
    X_PASSWORD - hasło do X

Przykład:
    export X_USERNAME="..."
    export X_PASSWORD="..."
    python news_scraper.py --ticker AAPL --company "Apple Inc" --hours 24

UWAGA: logowanie automatyczne do X może zostać zablokowane dodatkową
weryfikacją (captcha/SMS) - w takim wypadku zaloguj się ręcznie raz w
przeglądarce, wyeksportuj sesję i zapisz jako pattern_ml/.auth/x_state.json
(playwright storage_state), a skrypt użyje jej zamiast logowania hasłem.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

from src.x_auth import STORAGE_STATE_PATH, LoginError, ensure_logged_in
from src.x_scraper import fetch_fresh_mentions

CHROMIUM_PATH = "/opt/pw-browsers/chromium"

DEFAULT_ACCOUNTS = [
    "DeItaone",
    "unusual_whales",
    "FirstSquawk",
    "Reuters",
    "business",
]


def parse_args():
    p = argparse.ArgumentParser(description="Świeże wzmianki o instrumencie na X (Twitter)")
    p.add_argument("--ticker", required=True, help="Symbol giełdowy, np. AAPL")
    p.add_argument("--company", default=None, help="Pełna nazwa spółki, np. 'Apple Inc'")
    p.add_argument(
        "--accounts", default=",".join(DEFAULT_ACCOUNTS),
        help="Lista kont (bez @) rozdzielona przecinkami, monitorowanych pod kątem tickera",
    )
    p.add_argument("--hours", type=int, default=24, help="Maks. wiek posta w godzinach (świeżość)")
    p.add_argument("--max-scrolls", type=int, default=5, help="Ile razy doładować wyniki wyszukiwania")
    p.add_argument("--out", default="output", help="Katalog zapisu pliku wynikowego")
    p.add_argument("--headed", action="store_true", help="Uruchom przeglądarkę widocznie (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    accounts = [a.strip() for a in args.accounts.split(",") if a.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"news_{args.ticker.upper()}_{timestamp}.json"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed, executable_path=CHROMIUM_PATH)
        context = browser.new_context(
            storage_state=str(STORAGE_STATE_PATH) if STORAGE_STATE_PATH.exists() else None,
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        try:
            print("Sprawdzanie/wznawianie sesji logowania do X...")
            ensure_logged_in(context, page)
        except LoginError as exc:
            print(f"BŁĄD LOGOWANIA: {exc}", file=sys.stderr)
            browser.close()
            sys.exit(1)

        print(
            f"Szukam świeżych (<= {args.hours}h) wzmianek o {args.ticker} "
            f"(konta monitorowane: {', '.join(accounts)})..."
        )
        items = fetch_fresh_mentions(
            page, args.ticker, args.company, accounts, args.hours, max_scrolls=args.max_scrolls
        )
        browser.close()

    payload = {
        "ticker": args.ticker.upper(),
        "company": args.company,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freshness_hours": args.hours,
        "monitored_accounts": accounts,
        "count": len(items),
        "items": items,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nZnaleziono {len(items)} świeżych wzmianek. Zapisano do: {out_path}")
    for item in items[:5]:
        print(f"  [{item['published_at']}] {item['author_handle']}: {item['text'][:100]!r}")


if __name__ == "__main__":
    main()
