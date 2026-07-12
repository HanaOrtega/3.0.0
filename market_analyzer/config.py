# ============================================================
# CONFIGURATION
# ============================================================
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_FILE = os.environ.get("MARKET_DB_FILE", os.path.join(BASE_DIR, "market_news.db"))
HTML_DIR = os.environ.get("MARKET_HTML_DIR", os.path.join(BASE_DIR, "html_cache"))
REPORTS_DIR = os.environ.get("MARKET_REPORTS_DIR", os.path.join(BASE_DIR, "reports"))
DATA_DIR = os.path.join(BASE_DIR, "data")

OPML_FILE = os.environ.get("MARKET_OPML_FILE", os.path.join(DATA_DIR, "investing_feeds.opml"))
COMPANY_FILE = os.environ.get("MARKET_COMPANY_FILE", os.path.join(DATA_DIR, "company_map.json"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "180"))

MAX_RSS_ITEMS = int(os.environ.get("MARKET_MAX_RSS_ITEMS", "5"))
MIN_ARTICLE_CHARS = int(os.environ.get("MARKET_MIN_ARTICLE_CHARS", "300"))

# Backtest: an article needs to be at least this old before we can
# measure its 4h/24h price reaction.
BACKTEST_MIN_AGE_HOURS = int(os.environ.get("MARKET_BACKTEST_MIN_AGE_HOURS", "24"))

# Recommendation engine
RECO_LOOKBACK_DAYS = int(os.environ.get("MARKET_RECO_LOOKBACK_DAYS", "7"))
RECO_HALF_LIFE_HOURS = float(os.environ.get("MARKET_RECO_HALF_LIFE_HOURS", "48"))
RECO_BUY_THRESHOLD = float(os.environ.get("MARKET_RECO_BUY_THRESHOLD", "0.35"))
RECO_SELL_THRESHOLD = float(os.environ.get("MARKET_RECO_SELL_THRESHOLD", "-0.35"))
RECO_MIN_NEWS = int(os.environ.get("MARKET_RECO_MIN_NEWS", "2"))

DISCLAIMER = (
    "Ten raport jest generowany automatycznie na podstawie newsow i modelu AI. "
    "To NIE jest porada inwestycyjna - to narzedzie wspierajace decyzje. "
    "Zawsze zweryfikuj dane zrodlowe przed podjeciem decyzji inwestycyjnej."
)

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
