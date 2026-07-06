# Rozpoznawanie formacji świecowych z ML + analiza techniczna

Program pobiera dane rynkowe z **yfinance** (+ opcjonalnie świeże wzmianki z X
jako sentyment), liczy wskaźniki analizy technicznej, wykrywa klasyczne
formacje świecowe, a następnie trenuje stacking ensemble ML (RandomForest +
HistGradientBoosting), który przewiduje kierunek ceny w najbliższych świecach
oraz - regresorami kwantylowymi - prawdopodobny zakres przyszłej ceny.
Wszystko trafia na jeden wykres: formacje, wskaźniki, sugerowany kierunek
transakcji (**LONG / SHORT / NEUTRALNY**) i **prognozowany stożek ceny
rozciągnięty w przyszłość** za ostatnią świecą.

## GUI (platforma) — najszybszy start

Wszystkie trzy moduły projektu (sygnał ML + wykres, backtest, sentyment z X)
są dostępne w jednym interfejsie graficznym (`app.py`, [Streamlit](https://streamlit.io)).

### Wymagana wersja Pythona

**Python 3.11** (projekt testowany na 3.11.15; powinien też działać na 3.10 i
3.12 - jedyne twarde wymaganie to `scikit-learn>=1.3`, które wspiera 3.9-3.12).
Playwright (zakładka sentymentu) wymaga Pythona 3.9+.

### Instalacja i uruchomienie (terminal / dowolny system)

```bash
cd pattern_ml
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-gui.txt
# opcjonalnie, jeśli chcesz też zakładkę "Sentyment z X":
pip install -r requirements-news.txt

streamlit run app.py
```

Otworzy się przeglądarka pod `http://localhost:8501` z czterema zakładkami:
**Sygnał ML**, **Backtest**, **Sentyment z X**, **Informacje**. Wynik
zakładki "Sentyment z X" (po zaznaczeniu checkboxa) automatycznie zasila
zakładkę "Sygnał ML" jako dodatkowe cechy - bez ręcznego przenoszenia plików.

### Uruchomienie w PyCharm

1. **Otwórz folder `pattern_ml/` jako projekt** w PyCharm (File → Open).
2. **Ustaw interpreter Pythona 3.11**: File → Settings → Project → Python
   Interpreter → Add Interpreter → Add Local Interpreter → Virtualenv
   Environment → New, wskaż Python 3.11 jako bazowy interpreter, lokalizacja
   `pattern_ml/.venv`. Jeśli masz już utworzone `.venv` z kroku wyżej, wybierz
   zamiast tego "Existing environment" i wskaż `pattern_ml/.venv/bin/python`.
3. **Zainstaluj zależności** w zintegrowanym terminalu PyCharm (View → Tool
   Windows → Terminal - automatycznie użyje interpretera projektu):
   ```bash
   pip install -r requirements.txt -r requirements-gui.txt -r requirements-news.txt
   ```
4. **Uruchom aplikację** - najprościej z tego samego terminala:
   ```bash
   streamlit run app.py
   ```
   Alternatywnie, żeby mieć klikalny przycisk ▶ Run: Run → Edit Configurations
   → **+** → Python, w polu "Script path" wybierz interpreter modułu zamiast
   skryptu - ustaw "Module name" na `streamlit`, a w "Parameters" wpisz
   `run app.py`. Working directory ustaw na katalog `pattern_ml/`.
5. Do zakładki "Sentyment z X" ustaw zmienne środowiskowe **przed**
   uruchomieniem (w Run Configuration → Environment variables, albo
   `export X_USERNAME=... X_PASSWORD=...` w terminalu przed `streamlit run`).

## Instalacja (tylko CLI, bez GUI)

```bash
cd pattern_ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Użycie (CLI)

```bash
python main.py --ticker AAPL --period 2y --interval 1d --horizon 5

# ze sentymentem z X (najpierw uruchom news_scraper.py, patrz niżej):
python main.py --ticker AAPL --news-file output/news_AAPL_20260705T120000Z.json
```

Parametry:

| Flaga | Opis | Domyślnie |
|---|---|---|
| `--ticker` | Symbol giełdowy (np. `AAPL`, `BTC-USD`, `EURUSD=X`, `CDR.WA`) | `AAPL` |
| `--period` | Zakres pobieranych danych (`6mo`, `1y`, `2y`, `5y`...) | `2y` |
| `--interval` | Interwał świec (`1d`, `1h`, `1wk`...) | `1d` |
| `--horizon` | Liczba świec do przodu, dla której model przewiduje kierunek | `5` |
| `--atr-mult` | Próg (jako wielokrotność ATR%) odróżniający ruch od szumu rynkowego | `0.5` |
| `--min-confidence` | Próg pewności do wyciszenia słabych sygnałów do NEUTRALNY | auto (dobierany z CV) |
| `--last-n` | Liczba ostatnich świec pokazywanych na wykresie | `150` |
| `--save` | Ścieżka zapisu wykresu jako PNG | brak |
| `--no-show` | Nie otwieraj interaktywnego okna (przydatne np. przy `--save`) | wyłączone |
| `--news-file` | Plik JSON z `news_scraper.py` - dołącza sentyment z X jako dodatkowe cechy | brak |
| `--benchmark` | Indeks referencyjny do cech reżimu rynku (np. `^GSPC`, `^GDAXI`); pusty string wyłącza | `^GSPC` |

## Jak to działa

1. **`src/data.py`** — pobiera OHLCV z `yfinance`; ponawia pobieranie z
   wykładniczym backoffem przy chwilowym błędzie sieci, przycina `--period` do
   realnych limitów yfinance dla danych śróddziennych (inaczej zapytanie poza
   limitem po cichu zwraca puste/obcięte dane) i normalizuje strefę czasową
   indeksu.
2. **`src/quality.py`** — bramka jakości danych przed treningiem: twardy błąd
   przy stanowczo zbyt krótkiej historii, ostrzeżenia przy nieaktualnych danych
   (ticker może być wycofany z giełdy) i podejrzanie dużych przerwach w serii.
3. **`src/patterns.py`** — wykrywa formacje świecowe regułami opartymi o kształt
   świec (doji, młot, spadająca gwiazda, objęcie hossy/bessy, gwiazda poranna/
   wieczorna, trzej biali żołnierze/trzy czarne kruki, linia przebicia, zasłona
   ciemnej chmury) i buduje z nich skumulowany sygnał kierunkowy.
4. **`src/features.py`** — liczy wskaźniki analizy technicznej (SMA, EMA, RSI,
   MACD, Stochastic, Bollinger Bands, ATR, ADX, wolumen), kształt świecy oraz
   **cechy reżimu rynku** (fractional differencing log-ceny, wykładnik Hursta,
   entropia zwrotów - patrz niżej) i łączy je z formacjami świecowymi (+
   opcjonalnie sentymentem z X) w macierz cech dla modelu. Etykieta to kierunek
   ceny za `horizon` świec względem progu opartego o ATR (żeby odfiltrować
   szum). Wskaźniki nigdy nie są sztucznie uzupełniane (ffill/bfill) - wiersze
   z brakującymi danymi są odrzucane, z ostrzeżeniem, gdy to oznacza utratę
   dużej części małego zbioru.
5. **`src/sentiment.py`** — wczytuje wynik `news_scraper.py` i agreguje świeże
   wzmianki z X do dziennych cech (liczba wzmianek, zaangażowanie, prosta
   polaryzacja leksykonowa, dni od ostatniej wzmianki), dołączanych do macierzy
   cech z forward-fillem ograniczonym czasowo.
6. **`src/model.py`** — trenuje stacking ensemble (`StackingClassifier`) łączący
   `RandomForestClassifier` z `HistGradientBoostingClassifier` przez meta-model
   `LogisticRegression`, z walidacją krzyżową szeregu czasowego typu
   purged+embargo (bez przecieku danych z przyszłości - patrz niżej); zwraca
   prognozę (LONG/SHORT/NEUTRALNY) wraz z prawdopodobieństwami oraz
   **automatycznie dobrany próg pewności** maksymalizujący precyzję sygnałów
   kierunkowych (patrz niżej - "Adaptacyjny próg pewności"). Dodatkowo trenuje
   trzy regresory kwantylowe (`HistGradientBoostingRegressor`, percentyle
   10/50/90) przewidujące przyszłą stopę zwrotu - to one napędzają stożek
   prognozy ceny na wykresie.
7. **`src/plotting.py`** — rysuje wykres świecowy (`mplfinance`) z SMA/Bollinger,
   panelami RSI i MACD, wolumenem, znacznikami formacji (▲ bycze / ▼ niedźwiedzie),
   ramką z sygnałem ML i strzałką kierunku transakcji, a także **prognozowanym
   stożkiem ceny** (przerywana linia mediany + zacieniowany zakres P10-P90)
   rozciągniętym w przyszłość za ostatnią świecę - niepewność rośnie wraz
   z odległością w czasie (skalowanie `sqrt(t)`).
8. **`app.py` + `ui/`** — GUI (Streamlit) spinające punkty 1-7 oraz
   `news_scraper.py`/backtest w jednym interfejsie (patrz sekcja "GUI"
   wyżej) - `ui/signal_tab.py`, `ui/backtest_tab.py`, `ui/news_tab.py`
   wołają bezpośrednio te same funkcje z `src/`, którymi posługują się
   skrypty CLI (`main.py`, `backtest_run.py`, `news_scraper.py`) - jedna
   logika, dwa sposoby uruchomienia.

### Purged CV + embargo (poprawka przecieku danych)

Etykieta każdej próbki zależy od ceny `horizon` świec w przyszłość, więc zwykły
`TimeSeriesSplit` bez przerwy przecieka dane na granicy train/test. `gap=horizon`
usuwa ten podstawowy przeciek, ale nasze cechy mają też dłuższe "okno pamięci"
niż `horizon` (np. okno fractional differencing, patrz niżej, vs domyślny
`horizon=5`) - dlatego `src/model.py` dokłada dodatkowo **embargo**
(`MAX_INDICATOR_LOOKBACK=100` próbek odciętych z końca train foldu) oraz
dobiera liczbę foldów adaptacyjnie do rozmiaru danych (`choose_n_splits`),
zamiast sztywnego podziału, który przy krótkiej historii mógłby dać foldy
zbyt małe do sensownego treningu/oceny.

### Cechy reżimu rynku (fractional differencing, Hurst, entropia)

Trzy dodatkowe cechy w `src/features.py::_regime_features`, wzorowane na
`data/preprocessing_modules/feature_enrichment.py` (fractional differencing)
i `custom_features.py` (Hurst, entropia) z JuggleLab - dają modelowi surowiec
do rozróżnienia reżimu rynku, którego SMA/RSI/MACD same nie oddają:

- **`frac_diff_close`** - różnicowanie frakcyjne (Lopez de Prado) log-ceny:
  usuwa trend (potrzebne modelowi drzewiastemu, żeby nie "uczyć się" wprost
  poziomu ceny), ale w przeciwieństwie do zwykłych stóp zwrotu zachowuje część
  pamięci długoterminowej zamiast wyzerowywać ją przy każdym różnicowaniu.
- **`hurst`** - przybliżony wykładnik Hursta (skala log-log wariancji
  przyrostów): `<0.5` sugeruje reżim mean-reverting, `~0.5` błądzenie losowe,
  `>0.5` reżim trendujący. To uproszczony estymator (nie pełna analiza R/S) -
  bywa zaszumiony na krótkich oknach, ale jako cecha wejściowa dla modelu
  drzewiastego nie musi być idealnie skalibrowany, żeby nieść informację.
  Wydłuża wymaganą "rozgrzewkę" danych do ok. 100 świec.
- **`return_entropy`** - entropia Shannona rozkładu zwrotów w 20-świecowym
  oknie: niska = uporządkowany/kierunkowy ruch, wysoka = szum bez wyraźnego
  kierunku.

### Adaptacyjny próg pewności (minimalizacja strat na fałszywych sygnałach)

Zamiast zawsze podawać klasę o najwyższym prawdopodobieństwie (nawet gdy to
36% kontra 34% kontra 30% - ledwo różni się od zgadywania), `train_model`
dobiera próg pewności `recommended_confidence`, poniżej którego sygnał jest
wyciszany do NEUTRALNY. Próg jest dobierany na zbiorczych (pooled) out-of-fold
predykcjach ze wszystkich foldów walidacji krzyżowej, maksymalizując **precyzję
sygnałów kierunkowych (LONG/SHORT)**, nie ogólną trafność/F1 - inspirowane
`classification_thresholds.py` z JuggleLab, ale z celem dopasowanym do tego
projektu: skoro priorytetem jest "tracić jak najmniej", liczy się jakość
transakcji, na które model się faktycznie decyduje, a nie łapanie każdej
możliwej okazji. Próg ma wbudowany próg minimalnej "częstości sygnału"
(`min_signal_rate`), żeby nie wybrać progu tak wysokiego, że model nigdy nic
nie sygnalizuje. W `main.py` i GUI próg stosowany jest domyślnie automatycznie
(można go nadpisać ręcznie: `--min-confidence` / suwak w GUI).

### Cechy makro/reżimu całego rynku

Model oceniający pojedynczą spółkę (np. AAPL) wyłącznie na podstawie jej
własnych danych nie "wie", czy cały rynek jest w hossie, korekcie czy panice -
a formacja bycza w środku ogólnorynkowej wyprzedaży znaczy co innego niż ta
sama formacja w silnym trendzie wzrostowym. `src/macro.py` (wzorem
`data/macro.py::derive_market_regime_features` z JuggleLab) liczy z osobnego
indeksu referencyjnego (domyślnie `^GSPC`, S&P 500, konfigurowalne przez
`--benchmark`) zwrot 5d/21d, zmienność realizowaną 21d, obsunięcie od szczytu
i trend 63d - i dołącza je do macierzy cech głównego tickera (`mkt_*`). Puste
`--benchmark` wyłącza tę cechę.

### Selekcja cech (bez przecieku danych)

Po dodaniu sentymentu, reżimu rynku i cech statystycznych zbiór cech urósł do
40-50 kolumn - część bywa niemal stała albo silnie skorelowana z inną (np.
`return_5` z `return_10`), co zwiększa ryzyko przeuczenia bez realnej korzyści
informacyjnej. `src/feature_selection.py::select_features` (wzorem
`custom_features.py` z JuggleLab) odrzuca takie cechy automatycznie - **ale
tylko na podstawie samych cech**, nigdy etykiety. W JuggleLab filtr obejmował
też korelację cechy z targetem, co jest formą przecieku danych (dobór cech na
bazie tego, co dopiero mamy przewidzieć) - tu świadomie pominięte, spójnie z
resztą podejścia w tym projekcie (purged CV, embargo). Włączone domyślnie
(`auto_select_features=True` w `build_feature_matrix`).

### Adaptacyjne SL/TP ze stożka prognozy

W backteście (`src/backtest.py`, `--sizing-mode quantile`, domyślne) odległości
stop-loss/take-profit nie są już stałą wielokrotnością ATR, tylko wynikają z
rozrzutu prognozy kwantylowej P10/P90 (tej samej, która rysuje stożek na
wykresie w zakładce "Sygnał ML") - im szerszy stożek niepewności, tym szerszy
stop, i odwrotnie. Podłoga 0.5x ATR chroni przed zbyt ciasnym stopem przy
zdegenerowanym/wąskim stożku. `--sizing-mode atr` przywraca stare, czysto
ATR-owe zachowanie.

### Bogatsze metryki i zabezpieczenia walidacji krzyżowej

`train_model` (obok dotychczasowej trafności) liczy teraz na zbiorczych
out-of-fold predykcjach: **trafność zbalansowaną** i **MCC** (odporne na
przewagę liczebną klasy NEUTRALNY, w przeciwieństwie do zwykłej trafności),
**AUC-ROC** (multi-class, one-vs-rest) oraz - jeśli podano ciągłe stopy zwrotu
(`y_reg`) - **expectancy**: średni zwrot na transakcję kierunkową przy
zalecanym progu pewności (dodatnia = strategia w danych walidacyjnych
zarabiała więcej niż traciła na takich sygnałach). Foldy CV, w których dane
treningowe mają mniej niż 2 klasy (nie da się wytrenować klasyfikatora), są
pomijane zamiast po cichu psuć wynik - z licznikiem `skipped_folds`
widocznym w konsoli/GUI.

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
| `--train-window` | Rozmiar kroczącego okna treningowego | `300` |
| `--min-confidence` | Min. prawdopodobieństwo klasy, żeby wejść w pozycję | `0.40` |
| `--risk-pct` | Ryzyko na transakcję jako ułamek kapitału | `0.01` |
| `--sl-atr-mult` / `--tp-atr-mult` | Odległość stop-loss / take-profit jako wielokrotność ATR (podłoga w trybie `quantile`) | `1.5` / `2.5` |
| `--sizing-mode` | `quantile` (SL/TP ze stożka prognozy) albo `atr` (stałe wielokrotności ATR) | `quantile` |
| `--benchmark` | Indeks referencyjny do cech reżimu rynku; pusty string wyłącza | `^GSPC` |
| `--max-drawdown-halt` | Kill-switch: wstrzymaj nowe pozycje po tym obsunięciu kapitału | `0.25` |
| `--loss-streak-halt` | Kill-switch: wstrzymaj nowe pozycje po tylu stratnych transakcjach z rzędu | `5` |
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

### Kill-switch

`MLStrategy` wstrzymuje otwieranie **nowych** pozycji (istniejące dalej
zarządzane są normalnie przez SL/TP), gdy obsunięcie kapitału przekroczy
`--max-drawdown-halt` albo ostatnie `--loss-streak-halt` transakcji z rzędu
były stratne - to zabezpieczenie przed "upartym" handlem w reżimie rynkowym,
w którym model wyraźnie się myli, zamiast pozwolić stratom kumulować się bez
ograniczeń do końca backtestu.

### Uwaga o czasie działania

Retrenowanie ensemble (RandomForest + HistGradientBoosting) co `--retrain-every`
świec jest kosztowne obliczeniowo - backtest na kilkuset świecach może potrwać
od kilkudziesięciu sekund do kilku minut. Domyślny `--train-window` podniesiono
z 250 do 300 świec, żeby po nowej ~100-świecowej "rozgrzewce" cech reżimu rynku
(fractional differencing) nadal zostawało wystarczająco dużo próbek do treningu.
Zwiększ `--retrain-every` lub zmniejsz `--train-window`, żeby przyspieszyć
kosztem rzadszej aktualizacji modelu / mniejszej liczby dostępnych cech.

## Co zaczerpnięto z JuggleLab (i co celowo pominięto)

Kilka elementów tego projektu wzorowanych jest na analizie kodu
[JuggleLab](https://github.com/HanaOrtega/JuggleLab) - osobnego, znacznie
bardziej rozbudowanego pipeline'u ML do prognozowania cen (Temporal Fusion
Transformer, multi-stage training, tuner hiperparametrów). Przeniesiono
uproszczone, tanie w utrzymaniu wersje kilku pomysłów:

- **Sentyment z X** (`src/sentiment.py`) - wzorzec provider/join z
  `data/macro.py` + `data/sentiment.py`: osobne źródło danych agregowane do
  dziennych cech, dołączane forward-fillem z zerowym fallbackiem.
- **Odporność `fetch_ohlcv`** (`src/data.py`) - retry z backoffem, limit dni dla
  danych śróddziennych, normalizacja strefy czasowej - wzorem `data/fetching.py`.
- **Nigdy nie wypełniaj wskaźników sztucznie** (`src/features.py`) - wzorem
  `data/preprocessing_modules/data_cleaning.py`: NaN we wskaźnikach TA są
  odrzucane, nie interpolowane, z ostrzeżeniem przy dużej utracie danych.
- **Purged CV + embargo + adaptacyjne foldy** (`src/model.py`) - wzorem
  `pipeline/orchestrator_modules/cross_validation.py` i `cv_requirements.py`.
- **Stacking zamiast prostego uśredniania** (`src/model.py`) - wzorem
  `models/ensemble.py::train_stacking_meta_model` (tam `RidgeCV`, tu
  `LogisticRegression`, bo nasz problem to klasyfikacja, nie regresja).
- **Kill-switch w backteście** (`src/backtest.py`) - wzorem
  `pipeline/pnl_simulation/simulator.py`.
- **Pre-flight bramka jakości danych** (`src/quality.py`) - uproszczona wersja
  twardych sprawdzeń z `cv_requirements.py`/`cross_validation_validation.py`.
- **Cechy reżimu rynku** (`src/features.py::_regime_features`) - fractional
  differencing wzorem `feature_enrichment.py`, wykładnik Hursta i entropia
  zwrotów wzorem `custom_features.py`.
- **Adaptacyjny próg pewności** (`src/model.py::_tune_confidence_threshold`) -
  inspirowane `classification_thresholds.py`, ale z celem dopasowanym do tego
  projektu: maksymalizacja precyzji sygnałów kierunkowych (minimalizacja
  strat), nie ogólnego F1.
- **Cechy makro/reżimu rynku** (`src/macro.py`) - wzorem
  `data/macro.py::derive_market_regime_features`.
- **Selekcja cech** (`src/feature_selection.py`) - wzorem `custom_features.py`,
  z pominiętą (świadomie) korelacją cech z targetem, żeby nie wprowadzać
  przecieku danych.
- **Adaptacyjne SL/TP ze stożka prognozy** (`src/backtest.py::_risk_distances`) -
  wzorem `utils/trading.py::calculate_trade_levels`, ale oparte o nasze już
  istniejące regresory kwantylowe zamiast osobnej implementacji.
- **Bogatsze metryki CV + zabezpieczenia przed zdegenerowanymi foldami**
  (`src/model.py::train_model`) - wzorem `evaluation_classification.py`
  (balanced accuracy, MCC, AUC-ROC, guard na foldy z <2 klasami) oraz
  koncepcja "expectancy" z `pipeline/pnl_simulation/finance.py`, tu policzona
  tanio z już zebranych out-of-fold predykcji zamiast osobnej symulacji P&L.
- **Deklaratywna bramka jakości** (`src/quality.py::MetricRule`/`check_rules`) -
  lekka wersja `MetricRule`/`ResultValidator` z `quality_monitoring.py`.

Celowo pominięto: Temporal Fusion Transformer (`models/tft.py`) - deep learning
nieprzystający do sklearnowego ensemble; własny parser YAML i 3-warstwowy
system configów - zbędna złożoność dla projektu sterowanego flagami CLI;
integrację z MLflow/W&B/Neptune - print-y w konsoli wystarczają dla
jednoosobowego skryptu; rozbudowany system raportowania jakości danych
(Excel/JSONL, ~1100 linii) - dashboard dla zespołu, nie zabezpieczenie dla
pojedynczego uruchomienia. Część plików w JuggleLab (`cross_validation.py`,
`quality_monitoring.py`, `model_selection.py`) zawierała nierozwiązane
konflikty mergów w momencie analizy - potraktowano je jako punkt odniesienia
projektowy, nie kod do kopiowania 1:1.

## Uwaga

To narzędzie edukacyjne/analityczne, nie system automatycznego handlu.
Skuteczność modelu (raport `classification_report` i dokładność CV) zawsze
warto sprawdzić przed podjęciem jakiejkolwiek decyzji inwestycyjnej — rynki
finansowe są w dużej mierze losowe i żaden model nie daje gwarancji.
Automatyczne logowanie/scrapowanie X podlega regulaminowi platformy (X Terms
of Service) - używaj tego narzędzia na własnym koncie, w rozsądnych odstępach
czasu i wyłącznie do własnych celów analitycznych.
