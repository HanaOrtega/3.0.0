# news-summarizer

Program w Pythonie, ktory pobiera newsy na zadane tematy (np. spolki gieldowe)
z **wielu zrodel** (statyczne kanaly RSS, Google News RSS, opcjonalnie
NewsAPI i GNews, oraz generyczny scraping pelnej tresci kazdego artykulu z
dowolnej strony) i generuje **podsumowanie przy uzyciu lokalnego LLM**
(Ollama - dane nie sa wysylane do zadnego platnego API do podsumowan).
Podsumowanie kazdego artykulu i podsumowanie zbiorcze zawieraja tez ocene
LLM, jaki dana informacja moze miec wplyw na kierunek cen akcji (pozytywny /
negatywny / neutralny) - to automatyczna analiza sentymentu newsow, **nie
porada inwestycyjna**.

## Zrodla danych

| Zrodlo            | Wymaga klucza API | Uwagi |
|--------------------|--------------------|-------|
| Statyczne kanaly RSS | nie              | lista w `config/default.json` (`rssFeeds`) - dowolne kanaly |
| Google News RSS     | nie                | dynamiczne wyszukiwanie po `--query`, bez limitow |
| NewsAPI             | tak (`NEWSAPI_KEY`) | pomijane automatycznie, jesli brak klucza |
| GNews               | tak (`GNEWS_API_KEY`) | pomijane automatycznie, jesli brak klucza |
| Pelna tresc artykulu | nie               | generyczny scraping kazdego znalezionego linku (trafilatura) |

## Instalacja

```bash
cd news-summarizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Lokalny LLM (Ollama)

1. Zainstaluj Ollama: https://ollama.com/download
2. Pobierz model (raz): `ollama pull llama3` (albo inny, np. `llama3.2` lub `mistral`)
3. Upewnij sie, ze serwer dziala: `ollama serve` (na wielu systemach startuje
   automatycznie jako usluga)

### Opcjonalne klucze API (jesli chcesz uzywac NewsAPI / GNews)

```bash
export NEWSAPI_KEY="twoj_klucz"      # https://newsapi.org (darmowy plan dostepny)
export GNEWS_API_KEY="twoj_klucz"    # https://gnews.io (darmowy plan dostepny)
```

Bez tych zmiennych program po prostu pomija te dwa zrodla i korzysta z RSS +
Google News RSS + scrapingu tresci.

## Uzycie

Domyslnie program przetwarza **liste tematow z pliku konfiguracyjnego**
`config/default.json` (klucz `"queries"`, np. `["Google", "Apple"]`) i dla
kazdego z nich generuje osobny raport:

```bash
python main.py
```

Zeby przetworzyc tylko jeden, wybrany temat (ignorujac liste z configu), uzyj
`--query`:

```bash
python main.py --query "Google" --hours 72
python main.py --query "Google finanse" --lang pl --model llama3
python main.py --query "Tesla" --max-articles 20 --no-fulltext
```

Liste tematow edytujesz bezposrednio w `config/default.json`:

```json
"queries": ["Google", "Apple", "Tesla"],
```

| Flaga               | Opis                                                         | Domyslnie |
|----------------------|----------------------------------------------------------------|-----------|
| `--query`            | pojedynczy temat/haslo - nadpisuje liste `queries` z configu   | (lista z configu) |
| `--hours`             | ile godzin wstecz brac artykuly                                 | `72`      |
| `--lang`              | kod jezyka (np. `pl`, `en`)                                     | `pl`      |
| `--country`           | kod kraju dla Google News/GNews (np. `PL`, `US`)                | `PL`      |
| `--model`             | nazwa modelu Ollama                                             | `llama3`|
| `--ollama-host`       | adres serwera Ollama                                            | `http://localhost:11434` |
| `--max-articles`      | maks. liczba artykulow do podsumowania                          | `40`      |
| `--no-fulltext`       | nie scrapuj pelnej tresci - uzyj tylko opisu z RSS/API (szybsze)| wylaczone |
| `--no-rss` / `--no-google-news` / `--no-newsapi` / `--no-gnews` | wylacz dane zrodlo | wszystkie wlaczone |

Domyslne kanaly RSS i inne ustawienia mozna tez na stale zmienic w
`config/default.json`.

## Jak to dziala

1. Pobiera surowe wpisy (tytul, link, data, krotki opis) z kazdego wlaczonego
   zrodla.
2. Deduplikuje po URL/tytule i odrzuca wpisy starsze niz `--hours`.
3. Dla kazdego artykulu probuje pobrac pelna tresc ze strony (trafilatura) -
   jesli sie nie uda, uzywa opisu z RSS/API.
4. Kazdy artykul jest podsumowywany osobno przez lokalny LLM (Ollama) wraz z
   ocena mozliwego wplywu na kierunek ceny akcji, a nastepnie ze wszystkich
   podsumowan powstaje jedno podsumowanie zbiorcze z ogolna ocena sentymentu.
5. Dla kazdego tematu wynik trafia osobno do
   `output/news-<temat>-<timestamp>.md` (czytelny raport) oraz `.json`
   (surowe dane).

## Uwagi

- Jesli Ollama nie jest uruchomiona, program dziala dalej, ale bez podsumowan
  LLM - w raporcie znajdzie sie tylko oryginalny opis artykulu.
- Scraping pelnej tresci (`trafilatura`) czyta strukture HTML docelowych stron,
  wiec dla niektorych serwisow (paywalle, mocna ochrona anty-botowa) moze sie
  nie udac - wtedy uzywany jest opis z RSS/API jako fallback.
- Przy duzej liczbie artykulow generowanie podsumowan lokalnym LLM moze
  potrwac dlugo (zalezy od mocy komputera i wielkosci modelu) - zacznij od
  mniejszego `--max-articles`, jesli chcesz szybko przetestowac.
- Zbyt czeste/masowe scrapowanie wielu serwisow moze naruszac ich Regulamin -
  uzywaj rozsadnie i z umiarkowana czestotliwoscia.
