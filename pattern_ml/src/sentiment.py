"""Wpięcie świeżych wzmianek z X (news_scraper.py) do macierzy cech ML.

Wzorzec zaczerpnięty z data/macro.py + data/sentiment.py w projekcie JuggleLab:
osobne źródło danych agregowane do dziennych cech, dołączane przez `join` z
forward-fillem (posty nie pojawiają się co dnia) i prefiksem kolumn (żeby nie
kolidowały z istniejącymi cechami TA), z bezpiecznym fallbackiem do zera, gdy
brak jakichkolwiek wzmianek w danym dniu.

Sentyment liczony jest prostym, przejrzystym leksykonem słów kluczowych
(bez zależności NLP/transformerowych) ważonym zaangażowaniem (polubienia +
podania dalej) - wystarczające jako dodatkowy sygnał obok analizy technicznej,
nie próbuje być precyzyjnym klasyfikatorem sentymentu.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BULLISH_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "upgrade",
    "upgraded", "bullish", "buy", "outperform", "record high", "breakout",
    "strong", "growth", "profit", "raises guidance", "raise guidance",
    "wzrost", "hossa", "rekord", "przebicie", "rekomendacja kupuj",
}
BEARISH_WORDS = {
    "miss", "misses", "plunge", "plunges", "crash", "downgrade", "downgraded",
    "bearish", "sell", "underperform", "recall", "lawsuit", "investigation",
    "warning", "cuts guidance", "cut guidance", "weak", "loss", "bankruptcy",
    "spadek", "bessa", "rekomendacja sprzedaj", "ostrzeżenie",
}

_WORD_RE = re.compile(r"[a-ząćęłńóśźż]+", re.IGNORECASE)


def _post_polarity(text: str) -> int:
    words = set(_WORD_RE.findall(text.lower()))
    bull = len(words & BULLISH_WORDS)
    bear = len(words & BEARISH_WORDS)
    return int(np.sign(bull - bear))


def load_x_sentiment(json_path: str) -> pd.DataFrame:
    """Wczytuje plik JSON z news_scraper.py i agreguje wzmianki do dziennych cech:
    liczba wzmianek, łączne zaangażowanie, średnia polaryzacja (ważona
    zaangażowaniem), dzień ostatniej wzmianki."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z wynikami news_scraper.py: {json_path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        return pd.DataFrame(columns=["sent_mentions", "sent_engagement", "sent_polarity"])

    rows = []
    for item in items:
        published = pd.to_datetime(item["published_at"]).tz_localize(None)
        engagement = item.get("likes", 0) + item.get("retweets", 0) + item.get("replies", 0)
        polarity = _post_polarity(item.get("text", ""))
        rows.append({"date": published.normalize(), "engagement": engagement, "polarity": polarity})

    posts = pd.DataFrame(rows)
    weight = posts["engagement"] + 1  # +1 żeby posty bez zaangażowania też liczyły się w średniej

    daily = posts.groupby("date").apply(
        lambda g: pd.Series({
            "sent_mentions": len(g),
            "sent_engagement": g["engagement"].sum(),
            "sent_polarity": np.average(g["polarity"], weights=g["engagement"] + 1),
        }),
        include_groups=False,
    )
    daily.index.name = "Date"
    return daily.sort_index()


def merge_sentiment_features(
    features: pd.DataFrame,
    sentiment: pd.DataFrame,
    ffill_limit: int = 2,
) -> pd.DataFrame:
    """Dołącza dzienne cechy sentymentu do macierzy cech, z forward-fillem
    ograniczonym do `ffill_limit` dni (świeży news traci na aktualności) i
    zerowym fallbackiem, gdy w ogóle brak wzmianek.

    Uwaga: posty publikowane w weekendy/święta (gdy giełda jest zamknięta) nie
    mają odpowiednika w indeksie sesyjnym `features`. Zwykły `join` po dacie
    zgubiłby je całkowicie - dlatego forward-fill liczony jest na PEŁNYM
    kalendarzu (suma indeksu sesyjnego i dat sentymentu), a dopiero potem
    wynik przycinany jest z powrotem do dni sesyjnych z `features`.
    """
    out = features.copy()
    cols = ["sent_mentions", "sent_engagement", "sent_polarity"]

    if sentiment.empty:
        for col in cols:
            out[col] = 0.0
        out["sent_days_since_mention"] = 999.0
        return out

    full_index = out.index.union(sentiment.index).sort_values()
    daily = sentiment.reindex(full_index)
    raw_mentions = daily["sent_mentions"].fillna(0.0)  # przed ffillem - do liczenia "dni od ostatniej wzmianki"

    filled = daily[cols].ffill(limit=ffill_limit).fillna(0.0)

    mention_dates = full_index.to_series().where(raw_mentions > 0).ffill()
    days_since = (full_index.to_series() - mention_dates).dt.days
    filled["sent_days_since_mention"] = days_since.fillna(999).clip(upper=999)

    return out.join(filled)
