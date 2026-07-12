# ============================================================
# OPTIONAL FINBERT ENSEMBLE
#
# General-purpose LLMs frequently mis-score finance text because
# everyday-negative words ("liability", "exposure", "depreciation")
# are neutral or positive in a financial context. Domain-tuned models
# like ProsusAI/finbert are measurably better at financial sentiment
# (multiple published comparisons put the F1 gap at 4-5 points over
# generic sentiment tools). Used here as a second, independent opinion
# that recommend.py blends with the LLM's own sentiment - an ensemble,
# not a replacement, since the LLM still supplies direction/impact/
# event-type reasoning FinBERT doesn't do.
#
# This is optional: transformers + torch are heavy (~1GB+) and not
# required for the rest of the system to work. If unavailable, we
# degrade silently, same pattern as the playwright-stealth import.
# ============================================================
_pipeline = None
_load_attempted = False


def _get_pipeline():
    global _pipeline, _load_attempted
    if _load_attempted:
        return _pipeline
    _load_attempted = True
    try:
        from transformers import pipeline

        _pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        print("[i] FinBERT niedostepny (opcjonalne):", e)
        _pipeline = None
    return _pipeline


def is_available():
    return _get_pipeline() is not None


def finbert_sentiment(text):
    """Returns {'label': 'positive'|'negative'|'neutral', 'score': 0..1}
    or None if the optional dependency isn't installed."""
    pipe = _get_pipeline()
    if pipe is None:
        return None
    try:
        result = pipe(text[:512])[0]
        return {"label": result["label"].lower(), "score": float(result["score"])}
    except Exception as e:
        print("[!] FinBERT error:", e)
        return None
