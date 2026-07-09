"""Zapisuje raport zbiorczy w formacie Markdown."""
from datetime import datetime


def write_markdown_report(path, query, hours_back, overall_summary, articles):
    lines = [
        f"# Podsumowanie newsow: {query}",
        "",
        (
            f"_Wygenerowano: {datetime.now().isoformat(timespec='seconds')} "
            f"- okno czasowe: ostatnie {hours_back}h - liczba artykulow: {len(articles)}_"
        ),
        "",
        (
            "> **Uwaga:** oceny wplywu na kurs akcji ponizej sa automatycznie "
            "generowane przez lokalny LLM na podstawie tresci newsow i **nie "
            "stanowia porady inwestycyjnej**."
        ),
        "",
        "## Podsumowanie zbiorcze",
        "",
        overall_summary or "_(brak - LLM niedostepny lub brak artykulow z podsumowaniem)_",
        "",
        "## Artykuly",
        "",
    ]

    for a in articles:
        lines.append(f"### {a['title']}")
        meta = f"**Zrodlo:** {a['source']}"
        if a.get("published"):
            meta += f" | **Data:** {a['published']}"
        meta += f" | **Kanal danych:** {a.get('origin', '?')}"
        lines.append(meta)
        lines.append("")
        lines.append(a.get("summary") or a.get("description") or "_(brak podsumowania)_")
        lines.append("")
        lines.append(f"[Link do artykulu]({a['url']})")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
