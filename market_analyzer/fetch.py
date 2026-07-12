# ============================================================
# PLAYWRIGHT ARTICLE DOWNLOAD (with stealth)
# ============================================================
import hashlib
import os

from . import config

try:
    import playwright_stealth

    if hasattr(playwright_stealth, "stealth_sync"):
        _stealth_function = playwright_stealth.stealth_sync
    elif hasattr(playwright_stealth, "stealth"):
        _stealth_function = playwright_stealth.stealth
    else:
        _stealth_function = None
except Exception:
    _stealth_function = None


def apply_stealth(page):
    if _stealth_function:
        try:
            _stealth_function(page)
        except Exception:
            pass


def article_cache_path(url):
    file_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(config.HTML_DIR, file_hash + ".html")


def download_article(browser_context, url):
    filename = article_cache_path(url)
    if os.path.exists(filename):
        return filename

    page = browser_context.new_page()
    apply_stealth(page)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(2000)
        html = page.content()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        return filename
    except Exception as e:
        print("[!] Download error:", url, e)
        return None
    finally:
        page.close()
