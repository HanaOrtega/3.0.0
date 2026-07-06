"""
Rozpoznawanie formacji świecowych + ML + analiza techniczna dla wybranego instrumentu.

Przykład użycia:
    python main.py --ticker AAPL --period 2y --interval 1d --horizon 5
    python main.py --ticker AAPL --news-file output/news_AAPL_20260705T120000Z.json

Program:
  1. Pobiera dane OHLCV z yfinance (+ opcjonalnie świeże wzmianki z X, jeśli
     wskazano --news-file z news_scraper.py).
  2. Liczy wskaźniki analizy technicznej (SMA, EMA, RSI, MACD, Bollinger, ATR, ADX...).
  3. Wykrywa klasyczne formacje świecowe (młot, objęcie bessy/hossy, gwiazda poranna itd.).
  4. Trenuje stacking ensemble ML (RandomForest + HistGradientBoosting) przewidujący
     kierunek ceny za `horizon` świec, oraz regresory kwantylowe do prognozy ścieżki ceny.
  5. Rysuje wykres świecowy ze wskaźnikami, formacjami, sygnałem ML (LONG/SHORT/
     NEUTRALNY) oraz stożkiem prognozy ceny rozciągniętym w przyszłość.
"""

import argparse
import sys

import numpy as np

from src.data import fetch_ohlcv
from src.features import MAX_INDICATOR_LOOKBACK, build_feature_matrix
from src.macro import DEFAULT_BENCHMARK, derive_market_regime_features
from src.model import predict_latest, predict_price_path, train_model, train_quantile_models
from src.plotting import plot_chart
from src.quality import DataQualityError, validate_ohlcv
from src.sentiment import load_x_sentiment


def parse_args():
    p = argparse.ArgumentParser(description="Rozpoznawanie wzorców świecowych z ML")
    p.add_argument("--ticker", default="AAPL", help="Symbol giełdowy, np. AAPL, BTC-USD, EURUSD=X")
    p.add_argument("--period", default="2y", help="Zakres danych, np. 6mo, 1y, 2y, 5y")
    p.add_argument("--interval", default="1d", help="Interwał świec, np. 1d, 1h, 1wk")
    p.add_argument("--horizon", type=int, default=5, help="Ile świec do przodu przewiduje model")
    p.add_argument("--atr-mult", type=float, default=0.5, help="Próg klasyfikacji jako wielokrotność ATR%%")
    p.add_argument(
        "--min-confidence", type=float, default=None,
        help="Próg pewności do wyciszenia słabych sygnałów do NEUTRALNY (domyślnie: dobierany "
        "automatycznie na podstawie walidacji krzyżowej, maksymalizując precyzję sygnałów)",
    )
    p.add_argument("--last-n", type=int, default=150, help="Ile ostatnich świec pokazać na wykresie")
    p.add_argument("--save", default=None, help="Ścieżka do zapisu wykresu (np. wykres.png)")
    p.add_argument("--no-show", action="store_true", help="Nie otwieraj okna z wykresem (tylko zapis)")
    p.add_argument(
        "--news-file", default=None,
        help="Ścieżka do pliku JSON z news_scraper.py - dołącza sentyment z X jako dodatkowe cechy",
    )
    p.add_argument(
        "--benchmark", default=DEFAULT_BENCHMARK,
        help="Indeks referencyjny do cech reżimu rynku (np. ^GSPC, ^GDAXI); pusty string wyłącza",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Pobieranie danych dla {args.ticker} ({args.period}, {args.interval})...")
    df = fetch_ohlcv(args.ticker, period=args.period, interval=args.interval)
    print(f"Pobrano {len(df)} świec: {df.index[0].date()} -> {df.index[-1].date()}")

    try:
        quality = validate_ohlcv(df)
        quality.print_warnings()
    except DataQualityError as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        sys.exit(1)

    sentiment = None
    if args.news_file:
        print(f"Wczytywanie sentymentu z X: {args.news_file}...")
        sentiment = load_x_sentiment(args.news_file)
        print(f"Znaleziono dane sentymentu dla {len(sentiment)} dni.")

    market_regime = None
    if args.benchmark:
        try:
            print(f"Pobieranie indeksu referencyjnego {args.benchmark} (cechy reżimu rynku)...")
            benchmark_df = fetch_ohlcv(args.benchmark, period=args.period, interval=args.interval)
            market_regime = derive_market_regime_features(benchmark_df)
        except (ValueError, ConnectionError) as exc:
            print(f"UWAGA: nie udało się pobrać indeksu referencyjnego ({exc}) - pomijam cechy reżimu rynku.")

    print("Liczenie wskaźników technicznych i formacji świecowych...")
    features, pat, X, y, y_reg, feature_cols = build_feature_matrix(
        df, horizon=args.horizon, atr_mult=args.atr_mult, sentiment=sentiment, market_regime=market_regime
    )

    if len(X) < 100:
        print(
            f"UWAGA: tylko {len(X)} próbek treningowych - rozważ dłuższy --period "
            "dla bardziej wiarygodnego modelu.",
            file=sys.stderr,
        )

    print(f"Trenowanie modelu ML (stacking: RandomForest + HistGradientBoosting) na {len(X)} próbkach "
          f"(horyzont = {args.horizon} świec)...")
    result = train_model(X, y, gap=args.horizon, embargo=MAX_INDICATOR_LOOKBACK, y_reg=y_reg)

    if result.skipped_folds:
        print(f"UWAGA: pominięto {result.skipped_folds} fold(y) walidacji krzyżowej - za mało zróżnicowanych klas.")

    print(f"\nŚrednia trafność (walidacja krzyżowa szeregu czasowego): {result.cv_accuracy * 100:.1f}%")
    print(f"Trafność zbalansowana (odporna na przewagę klasy NEUTRALNY): {result.balanced_accuracy * 100:.1f}%")
    print(f"MCC (Matthews Correlation Coefficient, -1..1): {result.mcc:.2f}")
    if not np.isnan(result.auc_roc):
        print(f"AUC-ROC (multi-class, one-vs-rest): {result.auc_roc:.2f}")
    if not np.isnan(result.expectancy):
        print(
            f"Expectancy (śr. zwrot na transakcję kierunkową przy zalecanym progu pewności): "
            f"{result.expectancy * 100:+.2f}%"
        )
    print("\nRaport klasyfikacji (ostatni fold walidacyjny):")
    print(result.report)
    print("Najważniejsze cechy modelu:")
    print(result.feature_importances.head(10).to_string())

    if np.isnan(result.recommended_confidence_precision):
        print("\nZa mało danych walidacyjnych, żeby dobrać próg pewności - używam sygnału bez wyciszania.")
        min_confidence = args.min_confidence
    else:
        print(
            f"\nZalecany próg pewności (maks. precyzja sygnałów kierunkowych w walidacji): "
            f"{result.recommended_confidence * 100:.0f}% (szacowana precyzja: "
            f"{result.recommended_confidence_precision * 100:.0f}%)"
        )
        min_confidence = args.min_confidence if args.min_confidence is not None else result.recommended_confidence

    signal, proba, as_of = predict_latest(result, features, feature_cols, min_confidence=min_confidence)
    print(f"\n=== SYGNAŁ na {as_of.date()} dla {args.ticker}: {signal} ===")
    for label, p in proba.items():
        print(f"  {label}: {p * 100:.1f}%")

    print("\nTrenowanie regresorów kwantylowych (prognoza ścieżki ceny)...")
    quantile_models = train_quantile_models(X, y_reg)
    last_close = float(df["Close"].iloc[-1])
    quantile_prices = predict_price_path(quantile_models, features, feature_cols, last_close)
    print(
        f"Prognoza ceny za {args.horizon} świec: "
        f"{quantile_prices[0.1]:.2f} (P10) — {quantile_prices[0.5]:.2f} (mediana) — "
        f"{quantile_prices[0.9]:.2f} (P90), obecna cena: {last_close:.2f}"
    )

    print("\nGenerowanie wykresu...")
    fig = plot_chart(
        df, features, pat, args.ticker, signal, proba,
        quantile_prices=quantile_prices, horizon=args.horizon,
        last_n=args.last_n, save_path=args.save,
    )

    if args.save:
        print(f"Wykres zapisano w: {args.save}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
