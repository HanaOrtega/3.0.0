"""
Pełny walk-forward backtest sygnału ML (formacje świecowe + analiza techniczna)
z realistyczną egzekucją zleceń: prowizje, stop-loss/take-profit oparte o ATR,
wielkość pozycji oparta o ryzyko na transakcję. Model jest retrenowany co
`--retrain-every` świec na kroczącym oknie `--train-window`, widząc wyłącznie
dane do bieżącego momentu (bez przecieku z przyszłości).

Przykład:
    python backtest_run.py --ticker AAPL --period 3y --interval 1d
"""

import argparse
from pathlib import Path

from src.backtest import run_backtest
from src.data import fetch_ohlcv


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
    p.add_argument("--cash", type=float, default=10_000, help="Kapitał początkowy")
    p.add_argument("--commission", type=float, default=0.0007, help="Prowizja jako ułamek wartości transakcji")
    p.add_argument("--out", default="output", help="Katalog zapisu raportu HTML")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Pobieranie danych dla {args.ticker} ({args.period}, {args.interval})...")
    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    print(f"Pobrano {len(df)} świec: {df.index[0].date()} -> {df.index[-1].date()}")

    print(
        f"\nUruchamianie walk-forward backtestu: retrenowanie modelu co {args.retrain_every} "
        f"świec, okno treningowe {args.train_window} świec, horyzont decyzji {args.horizon} świec..."
    )
    bt, stats = run_backtest(
        df,
        horizon=args.horizon,
        atr_mult=args.atr_mult,
        retrain_every=args.retrain_every,
        train_window=args.train_window,
        min_confidence=args.min_confidence,
        risk_pct=args.risk_pct,
        sl_atr_mult=args.sl_atr_mult,
        tp_atr_mult=args.tp_atr_mult,
        cash=args.cash,
        commission=args.commission,
    )

    print("\n=== WYNIKI BACKTESTU ===")
    print(stats)

    trades = stats["_trades"]
    if len(trades):
        print("\nOstatnie transakcje:")
        cols = [c for c in ["EntryTime", "ExitTime", "Size", "EntryPrice", "ExitPrice", "PnL", "ReturnPct"] if c in trades.columns]
        print(trades[cols].tail(10).to_string(index=False))
    else:
        print("\nBrak transakcji w tym oknie czasowym - rozważ dłuższy --period, niższy --min-confidence.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"backtest_{args.ticker.upper()}.html"
    try:
        bt.plot(filename=str(html_path), open_browser=False)
        print(f"\nRaport (equity curve, drawdown, transakcje na wykresie) zapisany w: {html_path}")
    except Exception as exc:
        print(f"\nNie udało się wygenerować interaktywnego raportu HTML ({exc}), ale statystyki powyżej są kompletne.")


if __name__ == "__main__":
    main()
