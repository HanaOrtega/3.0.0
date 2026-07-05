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

## Świeże wzmianki o instrumencie z X (Twitter)

`news_scraper.py` loguje się do X (Playwright) i zbiera **wyłącznie świeże**
(domyślnie do 24h - `--hours`) posty wzmiankujące dany instrument, z zakładki
"Najnowsze" (nie "Najlepsze", żeby uniknąć starych, popularnych postów). Wynik
zapisywany jest do pliku JSON, jako uzupełniający sygnał sentymentu/newsflow
obok analizy technicznej z `main.py`.

### Instalacja

```bash
cd pattern_ml
source .venv/bin/activate
pip install -r requirements-news.txt
```

### Dane logowania

Skrypt **nigdy** nie przechowuje hasła w kodzie/repo - pobiera je wyłącznie
ze zmiennych środowiskowych:

```bash
export X_USERNAME="twój_login_lub_email"
export X_PASSWORD="twoje_hasło"
```

Po pierwszym udanym logowaniu sesja (cookies) zapisywana jest lokalnie w
`pattern_ml/.auth/x_state.json` (w `.gitignore`, nigdy nie trafia do repo) i
jest reużywana w kolejnych uruchomieniach, żeby nie logować się za każdym
razem - X często wymaga dodatkowej weryfikacji (captcha/SMS) przy logowaniu
z nowego adresu IP/kontenera. Jeśli logowanie automatyczne zostanie
zablokowane taką weryfikacją, zaloguj się raz ręcznie w zwykłej przeglądarce,
wyeksportuj sesję Playwright (`storage_state`) i podmień nią plik
`.auth/x_state.json`.

### Użycie

```bash
python news_scraper.py --ticker AAPL --company "Apple Inc" --hours 24
```

| Flaga | Opis | Domyślnie |
|---|---|---|
| `--ticker` | Symbol giełdowy (wymagane) | - |
| `--company` | Pełna nazwa spółki (poprawia trafność wyszukiwania) | brak |
| `--accounts` | Konta finansowe/newsowe monitorowane pod kątem tickera (bez `@`, po przecinku) | `DeItaone,unusual_whales,FirstSquawk,Reuters,business` |
| `--hours` | Maksymalny wiek posta w godzinach (filtr świeżości) | `24` |
| `--max-scrolls` | Ile razy doładować wyniki wyszukiwania (więcej = więcej postów, wolniej) | `5` |
| `--out` | Katalog zapisu pliku JSON | `output` |
| `--headed` | Uruchom przeglądarkę widocznie (debugowanie logowania) | wyłączone |

Wynik: `output/news_<TICKER>_<timestamp>.json` z listą postów (autor, treść,
URL, dokładny czas publikacji, liczba polubień/podań dalej/odpowiedzi) -
tylko z okna czasowego `--hours`, bez danych archiwalnych.

## Backtesting (walk-forward)

`backtest_run.py` sprawdza, czy sygnał ML faktycznie sprawdzałby się w praktyce -
symuluje handel na danych historycznych z realistyczną egzekucją (prowizje,
stop-loss/take-profit oparte o ATR, wielkość pozycji oparta o ryzyko na
transakcję) i liczy pełne statystyki strategii.

```bash
python backtest_run.py --ticker AAPL --period 3y --interval 1d
```

| Flaga | Opis | Domyślnie |
|---|---|---|
| `--ticker` / `--period` / `--interval` | Jak w `main.py` | `AAPL` / `3y` / `1d` |
| `--horizon` | Horyzont etykiety/decyzji w świecach | `5` |
| `--atr-mult` | Próg etykiety jako wielokrotność ATR% | `0.5` |
| `--retrain-every` | Co ile świec retrenować model | `20` |
| `--train-window` | Rozmiar kroczącego okna treningowego | `250` |
| `--min-confidence` | Min. prawdopodobieństwo klasy, żeby wejść w pozycję | `0.40` |
| `--risk-pct` | Ryzyko na transakcję jako ułamek kapitału | `0.01` |
| `--sl-atr-mult` / `--tp-atr-mult` | Odległość stop-loss / take-profit jako wielokrotność ATR | `1.5` / `2.5` |
| `--cash` | Kapitał początkowy | `10000` |
| `--commission` | Prowizja jako ułamek wartości transakcji | `0.0007` |
| `--out` | Katalog zapisu interaktywnego raportu HTML | `output` |

Wynik: pełne statystyki w konsoli (Return, Sharpe/Sortino/Calmar, max drawdown,
win rate, profit factor, liczba transakcji...) oraz interaktywny raport HTML
(equity curve, drawdown, transakcje na wykresie cenowym) w
`output/backtest_<TICKER>_ml.html`.

### Porównanie z prostymi strategiami bazowymi

```bash
python backtest_run.py --ticker AAPL --period 3y --compare-baselines
```

Dokłada trzy klasyczne, reguły-oparte strategie (`src/baselines.py`) i drukuje
tabelę porównawczą (Return, Sharpe, max drawdown, win rate, profit factor,
liczba transakcji) obok wyniku ML - **jeśli ensemble ML nie bije tych prostych
reguł, to znak, że jego "przewaga" jest iluzoryczna**, a nie że warto z niego
korzystać tylko dlatego, że jest bardziej wyrafinowany:

- **`turtle`** — klasyczny Donchian breakout: kup przy wybiciu ponad
  N-dniowe maksimum, zamknij pozycję przy zejściu poniżej N-dniowego minimum
  (`--turtle-window`, domyślnie 20).
- **`sma`** — crossover szybkiej/wolnej średniej kroczącej
  (`--sma-fast`/`--sma-slow`, domyślnie 10/30).
- **`contrarian`** — "kup dołek, sprzedaj górkę" z potwierdzeniem opóźnieniem
  (`--contrarian-delay`, domyślnie 3), żeby uniknąć whipsawów.

Logika tych trzech strategii wzorowana jest na najprostszych (nie-ML, nie-RL)
agentach z [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models)
(`agent/1.turtle-agent.ipynb`, `2.moving-average-agent.ipynb`,
`3.signal-rolling-agent.ipynb`) - z jedną poprawką: w oryginale okna liczone są
jako procent długości całego zbioru danych (np. "10% z len(df)"), co nie ma
sensu przy różnych okresach/interwałach; tutaj są to stałe, standardowe
parametry. Reszta tego repozytorium (18 wariantów LSTM/GRU/Transformer, 23
agentów RL/neuroewolucyjnych) celowo pominięta - większość tamtych przykładów
raportuje nierealistycznie wysoką skuteczność (np. "95.86% trafności") bez
walidacji walk-forward, co zwykle oznacza przeciek danych albo trenowanie/test
na tym samym oknie, a nie faktyczną przewagę predykcyjną.

### Jak to zbudowano (i dlaczego tak)

Silnik egzekucji zleceń i wzory na wszystkie metryki pochodzą z biblioteki
[`backtesting.py`](https://github.com/kernc/backtesting.py) (kernc/backtesting.py)
zamiast własnej, podatnej na błędy implementacji - to dojrzałe, szeroko używane
narzędzie w społeczności quant/Python. Wzorzec retrenowania modelu co N świec na
kroczącym oknie wewnątrz `Strategy.next()` odtwarza
[oficjalny przykład tej biblioteki "Trading with Machine Learning"](https://github.com/kernc/backtesting.py/blob/master/doc/examples/Trading%20with%20Machine%20Learning.py).

Kluczowa poprawka wzięta z literatury o walidacji modeli ML w finansach (koncepcja
*purgingu* z prac Marcosa Lópeza de Prado, *"Advances in Financial Machine
Learning"*): skoro etykieta każdej próbki zależy od ceny `horizon` świec w
przyszłość, to zwykły podział train/test bez przerwy powoduje przeciek danych na
granicy - ostatnie próbki treningowe "widzą" fragment okna testowego. Naprawiono
to na dwóch poziomach:
- w backteście: model w każdym momencie widzi tylko dane obcięte do bieżącej
  świecy (`self.data.df` w `backtesting.py` jest już tak obcięte), a
  `build_feature_matrix` samo odrzuca ostatnie `horizon` wierszy (brak jeszcze
  znanej etykiety);
- w treningu na żywo (`main.py` / `src/model.py`): `TimeSeriesSplit` dostał
  parametr `gap=horizon`, więc walidacja krzyżowa też pomija ten sam bufor
  między foldami.

### Uwaga o czasie działania

Retrenowanie ensemble (RandomForest + HistGradientBoosting) co `--retrain-every`
świec jest kosztowne obliczeniowo - backtest na kilkuset świecach może potrwać
od kilkudziesięciu sekund do kilku minut. Zwiększ `--retrain-every` lub zmniejsz
`--train-window`, żeby przyspieszyć kosztem rzadszej aktualizacji modelu.

## Uwaga

To narzędzie edukacyjne/analityczne, nie system automatycznego handlu.
Skuteczność modelu (raport `classification_report` i dokładność CV) zawsze
warto sprawdzić przed podjęciem jakiejkolwiek decyzji inwestycyjnej — rynki
finansowe są w dużej mierze losowe i żaden model nie daje gwarancji.
Automatyczne logowanie/scrapowanie X podlega regulaminowi platformy (X Terms
of Service) - używaj tego narzędzia na własnym koncie, w rozsądnych odstępach
czasu i wyłącznie do własnych celów analitycznych.
