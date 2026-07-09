"""
Pobranie pelnej tresci i daty publikacji artykulu przez scraping dowolnej
strony (bez API), uzywajac trafilatura (usuwa menu/reklamy/boilerplate,
zostawia tresc artykulu + metadane).
"""
import json

import trafilatura


def extract_article(url):
    """Zwraca dict {"text": str|None, "published": iso-str|None}.

    Data jest przydatna gl. dla zrodel typu 'listing page' (np. strona z
    newsami danej spolki), ktore same nie podaja daty przy linku - wtedy
    date bierzemy z metadanych strony docelowej artykulu.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"text": None, "published": None}
        raw = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            output_format="json",
            with_metadata=True,
        )
        if not raw:
            return {"text": None, "published": None}
        data = json.loads(raw)
        text = (data.get("text") or "").strip() or None
        published = data.get("date") or None
        return {"text": text, "published": published}
    except Exception:
        return {"text": None, "published": None}
