# ============================================================
# ARTICLE TEXT EXTRACTION
# ============================================================
from bs4 import BeautifulSoup

_COOKIE_WORDS = [
    "accept cookies",
    "privacy settings",
    "cookie policy",
    "choose to accept",
]

_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]


def extract_article_text(filename):
    if not filename:
        return ""
    try:
        with open(filename, encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "lxml")

        for tag in soup(_STRIP_TAGS):
            tag.decompose()

        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)
            if len(text) > 50:
                paragraphs.append(text)

        article = "\n".join(paragraphs)

        lowered = article.lower()
        if any(word in lowered for word in _COOKIE_WORDS):
            return ""

        return article[:30000]

    except Exception as e:
        print("[!] Extraction error:", e)
        return ""
