# ============================================================
# AI ANALYSIS (OLLAMA)
# ============================================================
import json
import re

import requests

from . import config

ANALYSIS_PROMPT = """Jestes profesjonalnym analitykiem rynku finansowego.
Przeanalizuj ponizsza wiadomosc i ocen jej wplyw na notowania wymienionych spolek/aktywow.

Wiadomosc:
{text}

Zwroc WYLACZNIE poprawny JSON (bez markdown, bez komentarzy) w formacie:
{{
"summary": "krotkie streszczenie po polsku (1-2 zdania)",
"sentiment": "positive|negative|neutral",
"direction": "UP|DOWN|NEUTRAL",
"impact_score": 0,
"confidence": 0,
"event_type": "earnings|merger_acquisition|regulatory|macro|analyst_rating|product|management|other",
"time_horizon_hours": 24,
"assets": "lista wspomnianych spolek/tickerow oddzielona przecinkami",
"reason": "krotkie uzasadnienie oceny"
}}
impact_score to liczba 0-10 (0 = brak wplywu na rynek, 10 = bardzo duzy wplyw).
confidence to liczba 0-1 (pewnosc modelu co do oceny).
time_horizon_hours to realistyczny czas w godzinach, w ktorym efekt powinien byc widoczny w cenie
(np. wyniki finansowe: 4-24h, fuzja/przejecie: 4h, decyzja regulatora: 24-72h, dane makro: 24h).
"""

_DEFAULT_ANALYSIS = {
    "summary": "",
    "sentiment": "neutral",
    "direction": "NEUTRAL",
    "impact_score": 0,
    "confidence": 0,
    "event_type": "other",
    "time_horizon_hours": 24,
    "assets": "",
    "reason": "",
}


def _extract_json(raw):
    """Ollama with format=json should return clean JSON, but models
    sometimes wrap it in markdown fences or add stray text - salvage
    the first {...} block if a direct parse fails."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def analyze_article(text):
    prompt = ANALYSIS_PROMPT.format(text=text)
    try:
        response = requests.post(
            config.OLLAMA_URL,
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=config.OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "{}")
        parsed = _extract_json(raw)
        if parsed is None:
            print("[!] Ollama zwrocil nieparsowalny JSON")
            return json.dumps(_DEFAULT_ANALYSIS)
        merged = {**_DEFAULT_ANALYSIS, **parsed}
        return json.dumps(merged)
    except Exception as e:
        print("[!] Ollama error:", e)
        return json.dumps(_DEFAULT_ANALYSIS)


def parse_analysis(analysis):
    try:
        return json.loads(analysis)
    except Exception:
        return dict(_DEFAULT_ANALYSIS)


def get_prediction_direction(analysis):
    return parse_analysis(analysis).get("direction", "NEUTRAL")


def get_impact_score(analysis):
    try:
        return float(parse_analysis(analysis).get("impact_score", 0))
    except (TypeError, ValueError):
        return 0.0


def get_confidence(analysis):
    try:
        return float(parse_analysis(analysis).get("confidence", 0))
    except (TypeError, ValueError):
        return 0.0


def get_sentiment(analysis):
    return parse_analysis(analysis).get("sentiment", "neutral")


def get_event_type(analysis):
    return parse_analysis(analysis).get("event_type", "other")


def get_time_horizon_hours(analysis):
    try:
        return float(parse_analysis(analysis).get("time_horizon_hours", 24) or 24)
    except (TypeError, ValueError):
        return 24.0


SENTIMENT_VALUE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def blended_sentiment_value(text, analysis):
    """Average of the LLM's own sentiment and FinBERT's (when available)
    into a single -1..1 value. Falls back to the LLM-only value if
    FinBERT isn't installed - see finbert.py."""
    from . import finbert

    llm_value = SENTIMENT_VALUE.get(get_sentiment(analysis), 0.0)

    fb = finbert.finbert_sentiment(text)
    if fb is None:
        return llm_value, False

    fb_value = SENTIMENT_VALUE.get(fb["label"], 0.0) * fb["score"]
    return (llm_value + fb_value) / 2.0, True
