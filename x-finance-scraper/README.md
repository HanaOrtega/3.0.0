# x-finance-scraper

Oddzielny, samodzielny program w **Pythonie** do pobierania postow z platformy
**X (Twitter) przez przegladarke** (Playwright + realny DOM x.com) - **bez
korzystania z oficjalnego API**. Domyslnie wyszukuje posty o zadanej
firmie/hasle (np. "Google") powiazane z tematyka finansowa (earnings, akcje,
przychody itp.) opublikowane w ciagu ostatnich **72 godzin**.

## Jak to dziala

1. X wymaga zalogowania, zeby przegladac wyniki wyszukiwania - dlatego program
   sklada sie z dwoch krokow: jednorazowego logowania (`login.py`, przegladarka
   z widocznym oknem) oraz wlasciwego scrapera (`scrape.py`, dziala headless,
   uzywa zapisanej sesji).
2. Scraper otwiera `x.com/search` z zakladka "Najnowsze" (`f=live`), scrolluje
   liste wynikow i odczytuje z DOM tresc, autora, link oraz znacznik czasu
   (`<time datetime>`) kazdego posta.
3. Posty starsze niz zadana liczba godzin sa odrzucane; scraper przestaje
   scrollowac, gdy kilka kolejnych partii wynikow jest juz spoza okna czasowego.
4. Wyniki zapisywane sa do `output/x-posts-<data>.json` i `.csv`.

## Instalacja

```bash
cd x-finance-scraper
python3 -m venv .venv && source .venv/bin/activate   # opcjonalnie
pip install -r requirements.txt
playwright install chromium   # tylko jesli przegladarka nie jest juz zainstalowana
```

## Krok 1: logowanie (jednorazowo, lokalnie na wlasnym komputerze)

```bash
python login.py
```

Otworzy sie widoczne okno przegladarki - zaloguj sie recznie na swoje konto X
(w razie potrzeby uzupelnij tez 2FA), a nastepnie wroc do terminala i nacisnij
Enter. Sesja zostanie zapisana do `auth/storage_state.json` i bedzie
wielokrotnie wykorzystywana przez scraper (dopoki nie wygasnie).

> Ten krok wymaga ekranu - uruchamiaj go na swoim komputerze, nie na serwerze
> bez GUI. Plik `auth/storage_state.json` mozna potem przeniesc na serwer.

## Krok 2: pobieranie postow

```bash
python scrape.py
```

Domyslnie: hasla `Google`, slowa kluczowe zwiazane z finansami, ostatnie 72h.

### Najczesciej uzywane opcje

```bash
python scrape.py --query "Google" --hours 72
python scrape.py --query "Tesla" --keywords "earnings,stock,revenue" --lang en
python scrape.py --query "Google" --strict          # odrzucaj posty bez slowa finansowego
python scrape.py --query "Google" --max-tweets 100 --max-scrolls 40
python scrape.py --query "Google" --headful          # pokaz okno przegladarki (wymaga ekranu)
```

| Flaga              | Opis                                                              | Domyslnie (z `config/default.json`) |
|---------------------|----------------------------------------------------------------------|---------------------------------------|
| `--query`           | glowne haslo/firma do wyszukania                                     | `Google`                              |
| `--keywords`        | lista slow finansowych oddzielonych przecinkami (OR)                  | patrz `config/default.json`           |
| `--hours`           | ile godzin wstecz pobierac posty                                      | `72`                                   |
| `--lang`            | kod jezyka X (np. `pl`, `en`) - opcjonalne                            | brak (wszystkie jezyki)               |
| `--strict`          | dodatkowy filtr: post musi zawierac haslo ORAZ slowo finansowe        | wylaczone                             |
| `--max-tweets`      | limit liczby zebranych postow                                        | `200`                                  |
| `--max-scrolls`     | limit liczby scrolli strony wynikow                                   | `60`                                   |
| `--headful`         | pokaz okno przegladarki zamiast trybu headless                        | headless                               |
| `--no-exclude-replies` | nie wykluczaj odpowiedzi (replies) z wynikow                       | wykluczone                             |

Domyslne slowa kluczowe i inne ustawienia mozna tez na stale zmienic w pliku
`config/default.json`.

## Wynik

Kazde uruchomienie tworzy w `output/`:
- `x-posts-<timestamp>.json` - pelne dane (tresc, autor, link, data, liczniki reakcji)
- `x-posts-<timestamp>.csv` - ta sama zawartosc w formacie CSV

## Uwagi

- To narzedzie odczytuje strukture DOM x.com, ktora **moze sie zmieniac** -
  jesli X zmieni interfejs, selektory w `scrape.py` moga wymagac aktualizacji.
- Zbyt czeste/agresywne scrapowanie moze skutkowac ograniczeniami na koncie
  (rate limiting, czasowa blokada). Zachowaj rozsadne odstepy miedzy uruchomieniami.
- Korzystanie z automatyzacji przegladarki do pobierania danych z X moze
  naruszac Regulamin platformy X - uzywaj tego narzedzia na wlasna
  odpowiedzialnosc, np. do celow analitycznych/badawczych na wlasnym koncie.
