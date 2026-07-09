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


def summarize_article(host, model, title, text, lang="pl"):
    body = (text or "")[:6000]
    lang_name = "polskim" if lang == "pl" else lang
    prompt = (
        f"Podsumuj ponizszy artykul w jezyku {lang_name} w 2-4 zdaniach, "
        "konkretnie i bez lania wody. Skup sie na faktach (liczby, decyzje, "
        "nazwy firm/osob), pomijaj wstep reklamowy i nawigacje strony.\n\n"
        f"Tytul: {title}\n\nTresc:\n{body}"
    )
    return _chat(host, model, prompt)


def summarize_digest(host, model, article_summaries, query, lang="pl"):
    lang_name = "polskim" if lang == "pl" else lang
    joined = "\n".join(f"- {a['title']} ({a['source']}): {a['summary']}" for a in article_summaries)
    prompt = (
        f"Ponizej jest lista {len(article_summaries)} podsumowan artykulow "
        f"na temat '{query}' z ostatnich godzin. Napisz zwiezle podsumowanie "
        f"zbiorcze (5-8 zdan) w jezyku {lang_name}: co sie dzieje, jakie sa "
        "najwazniejsze i powtarzajace sie watki, czy pojawiaja sie sprzeczne "
        "informacje miedzy zrodlami.\n\n"
        f"{joined}"
    )
    return _chat(host, model, prompt)
