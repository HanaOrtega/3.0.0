# ============================================================
# COMPANY / TICKER DETECTION
# ============================================================
import json
import os
import re

from . import config

_pattern_cache = {}


def load_company_map():
    if not os.path.exists(config.COMPANY_FILE):
        print("[!] Brak company_map.json")
        return {}
    try:
        with open(config.COMPANY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("[!] Company map error:", e)
        return {}


def _pattern_for(name):
    if name not in _pattern_cache:
        _pattern_cache[name] = re.compile(
            r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE
        )
    return _pattern_cache[name]


def detect_assets(text, company_map):
    """Word-boundary matching so short names (IBM, AMD, MS) don't match
    as substrings of unrelated words, and de-duplicates by ticker so a
    single article contributes only one row per company (e.g. Google
    and Alphabet both map to GOOGL)."""
    found = {}
    for company, data in company_map.items():
        if _pattern_for(company).search(text):
            ticker = data["ticker"]
            entry = found.setdefault(
                ticker,
                {
                    "company": set(),
                    "ticker": ticker,
                    "exchange": data.get("exchange", ""),
                    "sector": data.get("sector", ""),
                    "confidence": 0.9,
                },
            )
            entry["company"].add(company)

    results = []
    for entry in found.values():
        entry["company"] = ", ".join(sorted(entry["company"]))
        results.append(entry)
    return results
