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
"assets": "lista wspomnianych spolek/tickerow oddzielona przecinkami",
"reason": "krotkie uzasadnienie oceny"
}}
impact_score to liczba 0-10 (0 = brak wplywu na rynek, 10 = bardzo duzy wplyw).
confidence to liczba 0-1 (pewnosc modelu co do oceny).
"""

_DEFAULT_ANALYSIS = {
    "summary": "",
    "sentiment": "neutral",
    "direction": "NEUTRAL",
    "impact_score": 0,
    "confidence": 0,
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
