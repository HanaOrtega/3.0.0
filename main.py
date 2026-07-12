# ============================================================
# AI MARKET NEWS ANALYZER - CLI
#
# Uzycie:
#   python main.py scan        - pobiera i analizuje nowe newsy z RSS
#   python main.py backtest    - liczy skutecznosc AI na newsach starszych niz 24h
#   python main.py recommend   - generuje rekomendacje Kup/Sprzedaj/Trzymaj
#   python main.py digest      - generuje dzienny raport wplywu (--period weekly)
#   python main.py dashboard   - odpala interfejs Streamlit
#   python main.py all         - scan + backtest + recommend + digest
# ============================================================
import argparse
import sys

from market_analyzer.db import init_database


def main():
    parser = argparse.ArgumentParser(description="AI Market News Analyzer")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("scan", help="Pobierz i przeanalizuj nowe newsy")
    sub.add_parser("backtest", help="Policz skutecznosc AI na newsach ktore juz zdazyly zareagowac cenowo")
    sub.add_parser("recommend", help="Wygeneruj rekomendacje Kup/Sprzedaj/Trzymaj")

    digest_parser = sub.add_parser("digest", help="Wygeneruj raport wplywu newsow")
    digest_parser.add_argument("--period", choices=["daily", "weekly"], default="daily")

    sub.add_parser("dashboard", help="Uruchom dashboard Streamlit")
    sub.add_parser("all", help="scan + backtest + recommend + digest")

    args = parser.parse_args()

    init_database()

    if args.command == "scan":
        from market_analyzer.pipeline import run_pipeline
        run_pipeline()

    elif args.command == "backtest":
        from market_analyzer.backtest import run_pending_backtests
        run_pending_backtests()

    elif args.command == "recommend":
        from market_analyzer.recommend import generate_recommendations, print_recommendations
        results = generate_recommendations()
        print_recommendations(results)

    elif args.command == "digest":
        from market_analyzer.digest import build_digest
        md = build_digest(period=args.period)
        print(md)

    elif args.command == "dashboard":
        print("Uruchom zamiast tego: streamlit run market_analyzer/dashboard.py")
        sys.exit(1)

    elif args.command == "all":
        from market_analyzer.pipeline import run_pipeline
        from market_analyzer.backtest import run_pending_backtests
        from market_analyzer.recommend import generate_recommendations, print_recommendations
        from market_analyzer.digest import build_digest

        run_pipeline()
        run_pending_backtests()
        results = generate_recommendations()
        print_recommendations(results)
        build_digest(period="daily")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
