# Rozpoznawanie formacji świecowych z ML + analiza techniczna

Program pobiera dane rynkowe z **yfinance**, liczy wskaźniki analizy technicznej,
wykrywa klasyczne formacje świecowe, a następnie trenuje model ML (RandomForest),
który przewiduje kierunek ceny w najbliższych świecach. Wszystko trafia na jeden
wykres, na którym widać formacje, wskaźniki oraz sugerowany kierunek transakcji
(**LONG / SHORT / NEUTRALNY**).

## Instalacja

```bash
cd pattern_ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Użycie

```bash
python main.py --ticker AAPL --period 2y --interval 1d --horizon 5
```

Parametry:

| Flaga | Opis | Domyślnie |
|---|---|---|
| `--ticker` | Symbol giełdowy (np. `AAPL`, `BTC-USD`, `EURUSD=X`, `CDR.WA`) | `AAPL` |
| `--period` | Zakres pobieranych danych (`6mo`, `1y`, `2y`, `5y`...) | `2y` |
| `--interval` | Interwał świec (`1d`, `1h`, `1wk`...) | `1d` |
| `--horizon` | Liczba świec do przodu, dla której model przewiduje kierunek | `5` |
| `--atr-mult` | Próg (jako wielokrotność ATR%) odróżniający ruch od szumu rynkowego | `0.5` |
| `--last-n` | Liczba ostatnich świec pokazywanych na wykresie | `150` |
| `--save` | Ścieżka zapisu wykresu jako PNG | brak |
| `--no-show` | Nie otwieraj interaktywnego okna (przydatne np. przy `--save`) | wyłączone |

## Jak to działa

1. **`src/data.py`** — pobiera OHLCV z `yfinance`.
2. **`src/patterns.py`** — wykrywa formacje świecowe regułami opartymi o kształt
   świec (doji, młot, spadająca gwiazda, objęcie hossy/bessy, gwiazda poranna/
   wieczorna, trzej biali żołnierze/trzy czarne kruki, linia przebicia, zasłona
   ciemnej chmury) i buduje z nich skumulowany sygnał kierunkowy.
3. **`src/features.py`** — liczy wskaźniki analizy technicznej (SMA, EMA, RSI,
   MACD, Stochastic, Bollinger Bands, ATR, ADX, wolumen) i łączy je z formacjami
   świecowymi w macierz cech dla modelu. Etykieta to kierunek ceny za `horizon`
   świec względem progu opartego o ATR (żeby odfiltrować szum).
4. **`src/model.py`** — trenuje `RandomForestClassifier` z walidacją krzyżową
   szeregu czasowego (`TimeSeriesSplit`, bez przecieku danych z przyszłości) i
   zwraca prognozę (LONG/SHORT/NEUTRALNY) wraz z prawdopodobieństwami dla
   najnowszej świecy.
5. **`src/plotting.py`** — rysuje wykres świecowy (`mplfinance`) z SMA/Bollinger,
   panelami RSI i MACD, wolumenem, znacznikami formacji (▲ bycze / ▼ niedźwiedzie)
   oraz ramką z sygnałem ML i strzałką kierunku transakcji.

## Uwaga

To narzędzie edukacyjne/analityczne, nie system automatycznego handlu.
Skuteczność modelu (raport `classification_report` i dokładność CV) zawsze
warto sprawdzić przed podjęciem jakiejkolwiek decyzji inwestycyjnej — rynki
finansowe są w dużej mierze losowe i żaden model nie daje gwarancji.
