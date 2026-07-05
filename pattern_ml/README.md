# Rozpoznawanie formacji świecowych z ML + analiza techniczna

Program pobiera dane rynkowe z **yfinance**, liczy wskaźniki analizy technicznej,
wykrywa klasyczne formacje świecowe, a następnie trenuje ensemble ML
(RandomForest + HistGradientBoosting), który przewiduje kierunek ceny w
najbliższych świecach oraz - regresorami kwantylowymi - prawdopodobny zakres
przyszłej ceny. Wszystko trafia na jeden wykres: formacje, wskaźniki, sugerowany
kierunek transakcji (**LONG / SHORT / NEUTRALNY**) i **prognozowany stożek ceny
rozciągnięty w przyszłość** za ostatnią świecą.

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
4. **`src/model.py`** — trenuje miękki ensemble (`VotingClassifier`) łączący
   `RandomForestClassifier` z `HistGradientBoostingClassifier` (nowoczesny model
   boostingowy), z walidacją krzyżową szeregu czasowego (`TimeSeriesSplit`, bez
   przecieku danych z przyszłości); zwraca prognozę (LONG/SHORT/NEUTRALNY) wraz
   z prawdopodobieństwami. Dodatkowo trenuje trzy regresory kwantylowe
   (`HistGradientBoostingRegressor`, percentyle 10/50/90) przewidujące przyszłą
   stopę zwrotu - to one napędzają stożek prognozy ceny na wykresie.
5. **`src/plotting.py`** — rysuje wykres świecowy (`mplfinance`) z SMA/Bollinger,
   panelami RSI i MACD, wolumenem, znacznikami formacji (▲ bycze / ▼ niedźwiedzie),
   ramką z sygnałem ML i strzałką kierunku transakcji, a także **prognozowanym
   stożkiem ceny** (przerywana linia mediany + zacieniowany zakres P10-P90)
   rozciągniętym w przyszłość za ostatnią świecę - niepewność rośnie wraz
   z odległością w czasie (skalowanie `sqrt(t)`).

## Uwaga

To narzędzie edukacyjne/analityczne, nie system automatycznego handlu.
Skuteczność modelu (raport `classification_report` i dokładność CV) zawsze
warto sprawdzić przed podjęciem jakiejkolwiek decyzji inwestycyjnej — rynki
finansowe są w dużej mierze losowe i żaden model nie daje gwarancji.
