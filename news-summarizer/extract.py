"""
Pobranie pelnej tresci artykulu przez scraping dowolnej strony (bez API),
uzywajac trafilatura (usuwa menu/reklamy/boilerplate, zostawia tresc artykulu).
"""
import trafilatura


def extract_full_text(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return text.strip() if text else None
    except Exception:
        return None
