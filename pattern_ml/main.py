"""
Rozpoznawanie formacji świecowych + ML + analiza techniczna dla wybranego instrumentu.

Przykład użycia:
    python main.py --ticker AAPL --period 2y --interval 1d --horizon 5

Program:
  1. Pobiera dane OHLCV z yfinance.
  2. Liczy wskaźniki analizy technicznej (SMA, EMA, RSI, MACD, Bollinger, ATR, ADX...).
  3. Wykrywa klasyczne formacje świecowe (młot, objęcie bessy/hossy, gwiazda poranna itd.).
  4. Trenuje model ML (RandomForest) przewidujący kierunek ceny za `horizon` świec.
  5. Rysuje wykres świecowy ze wskaźnikami, formacjami i sygnałem ML (LONG/SHORT/NEUTRALNY).
"""

import argparse
import sys

from src.data import fetch_ohlcv
from src.features import build_feature_matrix
from src.model import predict_latest, train_model
from src.plotting import plot_chart


def parse_args():
    p = argparse.ArgumentParser(description="Rozpoznawanie wzorców świecowych z ML")
    p.add_argument("--ticker", default="AAPL", help="Symbol giełdowy, np. AAPL, BTC-USD, EURUSD=X")
    p.add_argument("--period", default="2y", help="Zakres danych, np. 6mo, 1y, 2y, 5y")
    p.add_argument("--interval", default="1d", help="Interwał świec, np. 1d, 1h, 1wk")
    p.add_argument("--horizon", type=int, default=5, help="Ile świec do przodu przewiduje model")
    p.add_argument("--atr-mult", type=float, default=0.5, help="Próg klasyfikacji jako wielokrotność ATR%%")
    p.add_argument("--last-n", type=int, default=150, help="Ile ostatnich świec pokazać na wykresie")
    p.add_argument("--save", default=None, help="Ścieżka do zapisu wykresu (np. wykres.png)")
    p.add_argument("--no-show", action="store_true", help="Nie otwieraj okna z wykresem (tylko zapis)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Pobieranie danych dla {args.ticker} ({args.period}, {args.interval})...")
    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    print(f"Pobrano {len(df)} świec: {df.index[0].date()} -> {df.index[-1].date()}")

    print("Liczenie wskaźników technicznych i formacji świecowych...")
    features, pat, X, y, feature_cols = build_feature_matrix(
        df, horizon=args.horizon, atr_mult=args.atr_mult
    )

    if len(X) < 100:
        print(
            f"UWAGA: tylko {len(X)} próbek treningowych - rozważ dłuższy --period "
            "dla bardziej wiarygodnego modelu.",
            file=sys.stderr,
        )

    print(f"Trenowanie modelu ML na {len(X)} próbkach (horyzont = {args.horizon} świec)...")
    result = train_model(X, y)

    print(f"\nŚrednia trafność (walidacja krzyżowa szeregu czasowego): {result.cv_accuracy * 100:.1f}%")
    print("\nRaport klasyfikacji (ostatni fold walidacyjny):")
    print(result.report)
    print("Najważniejsze cechy modelu:")
    print(result.feature_importances.head(10).to_string())

    signal, proba, as_of = predict_latest(result, features, feature_cols)
    print(f"\n=== SYGNAŁ na {as_of.date()} dla {args.ticker}: {signal} ===")
    for label, p in proba.items():
        print(f"  {label}: {p * 100:.1f}%")

    print("\nGenerowanie wykresu...")
    fig = plot_chart(
        df, features, pat, args.ticker, signal, proba,
        last_n=args.last_n, save_path=args.save,
    )

    if args.save:
        print(f"Wykres zapisano w: {args.save}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
