# AI Market News Analyzer

System do automatycznego zbierania newsow rynkowych (RSS), analizy ich wplywu
przez lokalny model AI (Ollama), wykrywania wspomnianych spolek, weryfikacji
trafnosci prognoz na realnych cenach (yfinance) oraz generowania rekomendacji
**Kup / Sprzedaj / Trzymaj** i dziennych **raportow wplywu**.

> **To narzedzie wspiera decyzje inwestycyjne, ale nie jest doradztwem
> inwestycyjnym.** Rekomendacje sa generowane automatycznie na podstawie
> newsow i modelu AI — zawsze zweryfikuj dane zrodlowe.

## Architektura

```
market_analyzer/
  config.py      - konfiguracja (sciezki, Ollama, progi rekomendacji)
  db.py          - schemat SQLite (news, assets, impact_backtest, recommendations, digests)
  feeds.py       - wczytywanie listy kanalow RSS z OPML
  fetch.py       - pobieranie artykulow przez Playwright (ze stealth)
  extract.py     - ekstrakcja tresci artykulu z HTML
  companies.py   - wykrywanie spolek/tickerow w tekscie (dopasowanie na granicach slow)
  llm.py         - analiza newsa przez Ollama (sentyment, kierunek, impact_score)
  pipeline.py    - pipeline zbierania: RSS -> pobranie -> ekstrakcja -> analiza -> zapis
  backtest.py    - weryfikacja trafnosci prognoz na realnych cenach (uruchamiane pozniej!)
  recommend.py   - silnik rekomendacji Kup/Sprzedaj/Trzymaj per ticker
  digest.py      - dzienny/tygodniowy raport wplywu newsow na rynek
  dashboard.py   - interfejs Streamlit
data/
  company_map.json      - mapa nazwa spolki -> ticker/gielda/sektor
  investing_feeds.opml  - lista kanalow RSS pogrupowana tematycznie
main.py          - CLI
```

## Kluczowa poprawka wzgledem oryginalnego skryptu

W oryginalnym `newsy_rrs.py` backtest byl liczony **od razu** po pobraniu
newsa, z `datetime.now()` jako punktem odniesienia — w efekcie "cena po 4h/24h"
nigdy nie mogla byc realna (artykul mial zero minut, nie 4-24h). W tym systemie
`backtest.py` to osobny krok (`python main.py backtest`), ktory znajduje newsy
starsze niz `BACKTEST_MIN_AGE_HOURS` (domyslnie 24h) i dopiero wtedy liczy
rzeczywisty ruch ceny od momentu publikacji newsa.

## Instalacja

```bash
pip install -r requirements.txt
playwright install chromium
```

Do analizy AI wymagany jest lokalnie uruchomiony [Ollama](https://ollama.com)
z pobranym modelem (domyslnie `llama3`):

```bash
ollama pull llama3
ollama serve
```

## Uzycie

```bash
# 1. Zbierz i przeanalizuj nowe newsy z RSS
python main.py scan

# 2. (uruchamiaj cyklicznie, np. co godzine) policz skutecznosc AI
#    na newsach ktore mialy juz czas zareagowac cenowo
python main.py backtest

# 3. Wygeneruj rekomendacje Kup/Sprzedaj/Trzymaj
python main.py recommend

# 4. Wygeneruj dzienny raport wplywu (zapisuje sie do reports/ i do bazy)
python main.py digest --period daily

# Albo wszystko na raz:
python main.py all

# Dashboard:
streamlit run market_analyzer/dashboard.py
```

Sugerowany harmonogram (np. cron): `scan` co 30-60 min, `backtest` co godzine,
`recommend` + `digest` raz dziennie.

## Jak dziala rekomendacja Kup/Sprzedaj

Dla kazdego tickera z newsami w ostatnich `RECO_LOOKBACK_DAYS` dniach (domyslnie 7):

1. Kazdy news dostaje sygnal `(sentyment + kierunek) / 2` w zakresie -1..1.
2. Sygnal jest wazony przez: `impact_score` z AI, pewnosc AI oraz **swiezosc**
   (exponential decay z polowicznym okresem `RECO_HALF_LIFE_HOURS`, domyslnie 48h).
3. Liczona jest srednia wazona -> `score` tickera.
4. `score >= 0.35` -> **BUY**, `score <= -0.35` -> **SELL**, w przeciwnym razie **HOLD**
   (wymagana minimalna liczba newsow: `RECO_MIN_NEWS`).
5. `confidence` uwzglednia zgodnosc newsow miedzy soba, historyczna skutecznosc
   AI na danym tickerze (z tabeli `impact_backtest`) oraz liczbe newsow.

Wszystkie progi sa konfigurowalne przez zmienne srodowiskowe (patrz `market_analyzer/config.py`).

## Konfiguracja (zmienne srodowiskowe)

| Zmienna | Domyslnie | Opis |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | endpoint Ollama |
| `OLLAMA_MODEL` | `llama3` | model uzywany do analizy |
| `MARKET_MAX_RSS_ITEMS` | `5` | ile najnowszych wpisow z kazdego kanalu RSS przetwarzac |
| `MARKET_BACKTEST_MIN_AGE_HOURS` | `24` | jak stary musi byc news, zeby policzyc backtest |
| `MARKET_RECO_LOOKBACK_DAYS` | `7` | okno newsow branych pod uwage w rekomendacji |
| `MARKET_RECO_BUY_THRESHOLD` / `MARKET_RECO_SELL_THRESHOLD` | `0.35` / `-0.35` | progi decyzji |
