"""
Podsumowania przez lokalny LLM uruchomiony w Ollama (http://localhost:11434).
Wymaga: zainstalowanej Ollama, uruchomionego `ollama serve` i pobranego modelu
(np. `ollama pull llama3.1`).
"""
import requests

DEFAULT_TIMEOUT = 120


class OllamaError(RuntimeError):
    pass


def check_ollama_available(host):
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def _chat(host, model, prompt, timeout=DEFAULT_TIMEOUT):
    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message", {}).get("content") or "").strip()
    except requests.exceptions.RequestException as e:
        raise OllamaError(str(e)) from e


def summarize_article(host, model, title, text, query, lang="pl"):
    body = (text or "")[:6000]
    lang_name = "polskim" if lang == "pl" else lang
    prompt = (
        f"Podsumuj ponizszy artykul w jezyku {lang_name} w 2-4 zdaniach, "
        "konkretnie i bez lania wody. Skup sie na faktach (liczby, decyzje, "
        "nazwy firm/osob), pomijaj wstep reklamowy i nawigacje strony. "
        f"Na koniec dodaj jedno zdanie zaczynajace sie od 'Wplyw na kurs akcji "
        f"{query}:' oceniajace, czy ta informacja jest raczej pozytywna, "
        "negatywna czy neutralna dla kierunku ceny akcji, z krotkim "
        "uzasadnieniem. Jesli artykul nie dotyczy bezposrednio finansow/gieldy, "
        "napisz 'Wplyw na kurs akcji: brak bezposredniego zwiazku'.\n\n"
        f"Tytul: {title}\n\nTresc:\n{body}"
    )
    return _chat(host, model, prompt)


def summarize_digest(host, model, article_summaries, query, lang="pl"):
    lang_name = "polskim" if lang == "pl" else lang
    joined = "\n".join(f"- {a['title']} ({a['source']}): {a['summary']}" for a in article_summaries)
    prompt = (
        f"Ponizej jest lista {len(article_summaries)} podsumowan artykulow "
        f"(kazde zawiera tez ocene wplywu na kurs akcji) na temat '{query}' "
        f"z ostatnich godzin. Napisz w jezyku {lang_name}:\n"
        "1) Zwiezle podsumowanie zbiorcze (5-8 zdan): co sie dzieje, jakie sa "
        "najwazniejsze i powtarzajace sie watki, czy pojawiaja sie sprzeczne "
        "informacje miedzy zrodlami.\n"
        f"2) Osobny akapit 'Ocena wplywu na kierunek cen akcji {query}:' - na "
        "podstawie wszystkich powyzszych ocen okresl ogolny sentyment "
        "(pozytywny / negatywny / neutralny / mieszany) i krotko uzasadnij "
        "(2-3 zdania). Wyraznie zaznacz, ze to automatyczna analiza newsow, a "
        "nie porada inwestycyjna.\n\n"
        f"{joined}"
    )
    return _chat(host, model, prompt)
