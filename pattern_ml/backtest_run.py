"""
Pełny walk-forward backtest sygnału ML (formacje świecowe + analiza techniczna)
z realistyczną egzekucją zleceń: prowizje, stop-loss/take-profit oparte o ATR,
wielkość pozycji oparta o ryzyko na transakcję. Model jest retrenowany co
`--retrain-every` świec na kroczącym oknie `--train-window`, widząc wyłącznie
dane do bieżącego momentu (bez przecieku z przyszłości).

Z flagą --compare-baselines dokłada trzy klasyczne, reguły-oparte strategie
(Donchian/turtle breakout, SMA crossover, kontrariańska z opóźnieniem) jako
punkt odniesienia - żeby sprawdzić, czy model ML faktycznie daje przewagę
nad prostymi regułami, a nie tylko wygląda dobrze w izolacji.

Przykład:
    python backtest_run.py --ticker AAPL --period 3y --interval 1d --compare-baselines
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest
from src.baselines import BASELINES, run_baseline_backtest
from src.data import fetch_ohlcv
from src.quality import DataQualityError, validate_ohlcv

SUMMARY_COLS = [
    "Return [%]", "Buy & Hold Return [%]", "Sharpe Ratio", "Max. Drawdown [%]",
    "Win Rate [%]", "Profit Factor", "# Trades",
]


def parse_args():
    p = argparse.ArgumentParser(description="Walk-forward backtest sygnału ML")
    p.add_argument("--ticker", default="AAPL", help="Symbol giełdowy")
    p.add_argument("--period", default="3y", help="Zakres danych historycznych")
    p.add_argument("--interval", default="1d", help="Interwał świec")
    p.add_argument("--horizon", type=int, default=5, help="Horyzont etykiety/decyzji (świece)")
    p.add_argument("--atr-mult", type=float, default=0.5, help="Próg etykiety jako wielokrotność ATR%%")
    p.add_argument("--retrain-every", type=int, default=20, help="Co ile świec retrenować model")
    p.add_argument("--train-window", type=int, default=250, help="Rozmiar kroczącego okna treningowego")
    p.add_argument("--min-confidence", type=float, default=0.40, help="Min. prawdopodobieństwo do wejścia w pozycję")
    p.add_argument("--risk-pct", type=float, default=0.01, help="Ryzyko na transakcję jako ułamek kapitału")
    p.add_argument("--sl-atr-mult", type=float, default=1.5, help="Odległość stop-loss jako wielokrotność ATR")
    p.add_argument("--tp-atr-mult", type=float, default=2.5, help="Odległość take-profit jako wielokrotność ATR")
    p.add_argument("--max-drawdown-halt", type=float, default=0.25, help="Kill-switch: wstrzymaj nowe pozycje po tej wielkości obsunięcia kapitału")
    p.add_argument("--loss-streak-halt", type=int, default=5, help="Kill-switch: wstrzymaj nowe pozycje po tylu stratnych transakcjach z rzędu")
    p.add_argument("--cash", type=float, default=10_000, help="Kapitał początkowy")
    p.add_argument("--commission", type=float, default=0.0007, help="Prowizja jako ułamek wartości transakcji")
    p.add_argument("--out", default="output", help="Katalog zapisu raportów HTML")
    p.add_argument(
        "--compare-baselines", action="store_true",
        help="Dodatkowo uruchom klasyczne strategie bazowe (turtle/sma/contrarian) do porównania",
    )
    p.add_argument("--turtle-window", type=int, default=20, help="Okno breakoutu Donchian (turtle)")
    p.add_argument("--sma-fast", type=int, default=10, help="Szybka SMA (crossover)")
    p.add_argument("--sma-slow", type=int, default=30, help="Wolna SMA (crossover)")
    p.add_argument("--contrarian-delay", type=int, default=3, help="Opóźnienie potwierdzenia (kontrariańska)")
    return p.parse_args()


def _print_trades(stats):
    trades = stats["_trades"]
    if len(trades):
        print("\nOstatnie transakcje:")
        cols = [c for c in ["EntryTime", "ExitTime", "Size", "EntryPrice", "ExitPrice", "PnL", "ReturnPct"] if c in trades.columns]
        print(trades[cols].tail(10).to_string(index=False))
    else:
        print("\nBrak transakcji w tym oknie czasowym.")


def _save_report(bt, out_dir: Path, ticker: str, label: str):
    html_path = out_dir / f"backtest_{ticker.upper()}_{label}.html"
    try:
        bt.plot(filename=str(html_path), open_browser=False)
        print(f"Raport zapisany w: {html_path}")
    except Exception as exc:
        print(f"Nie udało się wygenerować raportu HTML dla '{label}' ({exc}).")


def main():
    args = parse_args()

    print(f"Pobieranie danych dla {args.ticker} ({args.period}, {args.interval})...")
    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    print(f"Pobrano {len(df)} świec: {df.index[0].date()} -> {df.index[-1].date()}")

    try:
        quality = validate_ohlcv(df, min_rows=150)  # backtest potrzebuje więcej historii niż pojedynczy sygnał
        quality.print_warnings()
    except DataQualityError as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print(
        f"\n=== Sygnał ML: retrenowanie co {args.retrain_every} świec, okno treningowe "
        f"{args.train_window} świec, horyzont decyzji {args.horizon} świec ==="
    )
    bt_ml, stats_ml = run_backtest(
        df,
        horizon=args.horizon,
        atr_mult=args.atr_mult,
        retrain_every=args.retrain_every,
        train_window=args.train_window,
        min_confidence=args.min_confidence,
        risk_pct=args.risk_pct,
        sl_atr_mult=args.sl_atr_mult,
        tp_atr_mult=args.tp_atr_mult,
        max_drawdown_halt=args.max_drawdown_halt,
        loss_streak_halt=args.loss_streak_halt,
        cash=args.cash,
        commission=args.commission,
    )
    print(stats_ml)
    _print_trades(stats_ml)
    _save_report(bt_ml, out_dir, args.ticker, "ml")
    results["ML (ensemble)"] = stats_ml

    if args.compare_baselines:
        baseline_params = {
            "turtle": {"window": args.turtle_window},
            "sma": {"fast": args.sma_fast, "slow": args.sma_slow},
            "contrarian": {"delay": args.contrarian_delay},
        }
        for name in BASELINES:
            print(f"\n=== Strategia bazowa: {name} ===")
            bt_base, stats_base = run_baseline_backtest(
                df, name, cash=args.cash, commission=args.commission, **baseline_params[name]
            )
            print(stats_base)
            _save_report(bt_base, out_dir, args.ticker, name)
            results[name] = stats_base

        print("\n=== PORÓWNANIE STRATEGII ===")
        summary = {label: {col: stats.get(col) for col in SUMMARY_COLS} for label, stats in results.items()}
        print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
