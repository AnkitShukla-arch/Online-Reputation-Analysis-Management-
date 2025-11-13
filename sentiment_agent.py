"""
Sentiment / Analysis Agent (fixed)
- FastAPI app accepts batches of mentions and returns sentiment, summary, reputation,
  trending keywords, alerts, and suggested responses (optional OpenAI).
Run: python sentiment_agent.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import math
import logging
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Optional OpenAI usage (safe import)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

# Ensure NLTK data (download if missing)
nltk_packages = ["vader_lexicon", "punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        if pkg == "vader_lexicon":
            nltk.data.find("sentiment/vader_lexicon")
        else:
            nltk.data.find(pkg)
    except Exception:
        nltk.download(pkg)

sia = SentimentIntensityAnalyzer()

app = FastAPI(title="Sentiment Agent", version="1.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_agent")

# ---------------------------
# Configuration
# ---------------------------
NEGATIVE_ALERT_RATIO = float(os.getenv("NEG_ALERT_RATIO", 0.30))
NEGATIVE_ALERT_MIN_MENTIONS = int(os.getenv("NEG_ALERT_MIN_MENTIONS", 10))
TOP_K_KEYWORDS = int(os.getenv("TOP_K_KEYWORDS", 10))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

if OPENAI_API_KEY and OPENAI_AVAILABLE and openai is not None:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception:
        logger.warning("Could not set OpenAI API key on the openai module.")
elif OPENAI_API_KEY and not OPENAI_AVAILABLE:
    logger.warning("OPENAI_API_KEY provided, but openai package is not installed or failed to import.")


# ---------------------------
# Data models
# ---------------------------
class Mention(BaseModel):
    id: Optional[str] = None
    platform: Optional[str] = None
    author: Optional[str] = None
    text: str
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProcessRequest(BaseModel):
    brand: Optional[str] = None
    mentions: List[Mention]
    historical_negative_ratio: Optional[float] = None
    historical_window_size: Optional[int] = None


# ---------------------------
# Helper functions
# ---------------------------
def analyze_sentiment(text: str) -> Dict[str, Any]:
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"scores": scores, "label": label, "compound": compound}


def extract_trending_keywords(texts: List[str], top_k: int = TOP_K_KEYWORDS) -> List[Dict[str, Any]]:
    """
    TF-IDF across docs -> sum column-wise -> get top features.
    Implemented in a type-safe way for sparse matrices so .sum doesn't cause type-checker issues.
    """
    if not texts:
        return []

    docs = [t.lower() for t in texts]
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    try:
        X = vect.fit_transform(docs)  # X is sparse matrix (CSR)
    except ValueError:
        return []

    # X.sum(axis=0) returns a (1, n_features) matrix-like object for sparse matrices.
    # Convert safely to a 1-D numpy array:
    col_sums = np.asarray(X.sum(axis=0)).reshape(-1)  # safe conversion to 1D
    # Alternatively: col_sums = np.ravel(X.sum(axis=0).A)  # if .A available for sparse
    features = np.array(vect.get_feature_names_out())
    # sort descending and pick top_k
    top_idx = np.argsort(col_sums)[::-1][:top_k]
    keywords = [{"keyword": features[i], "score": float(col_sums[i])} for i in top_idx if col_sums[i] > 0]
    return keywords


def compute_reputation_score(positive_count: int, neutral_count: int, negative_count: int) -> float:
    total = positive_count + neutral_count + negative_count
    if total == 0:
        return 50.0
    pos_pct = positive_count / total
    neg_pct = negative_count / total
    diff = pos_pct - neg_pct
    score = 50 + (diff * 50)
    return round(max(0.0, min(100.0, score)), 2)


def detect_critical_alerts(
    positive_count: int,
    neutral_count: int,
    negative_count: int,
    historical_negative_ratio: Optional[float],
    window_size: Optional[int],
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    total = positive_count + neutral_count + negative_count
    if total == 0:
        return alerts
    neg_ratio = negative_count / total

    if total >= NEGATIVE_ALERT_MIN_MENTIONS and neg_ratio >= NEGATIVE_ALERT_RATIO:
        alerts.append({
            "type": "high_negative_ratio",
            "message": f"High negative ratio: {neg_ratio:.2%} over {total} mentions (threshold {NEGATIVE_ALERT_RATIO:.2%})",
            "neg_ratio": neg_ratio,
            "total_mentions": total
        })

    if historical_negative_ratio is not None and window_size is not None and window_size > 0:
        if historical_negative_ratio == 0 and neg_ratio > 0:
            fold_change = float('inf')
        else:
            fold_change = neg_ratio / max(1e-6, historical_negative_ratio)
        if fold_change >= 2.0 and neg_ratio >= 0.05:
            alerts.append({
                "type": "negative_spike",
                "message": f"Negative mentions spiked {fold_change:.2f}x vs historical ({historical_negative_ratio:.2%}).",
                "fold_change": fold_change,
                "neg_ratio": neg_ratio
            })
    return alerts


async def generate_response_draft_openai(brand: Optional[str], mention_text: str, author: Optional[str]) -> str:
    """
    Generates a reply draft. Uses OpenAI only if available and model/method exist.
    Otherwise falls back to a templated reply.
    """
    prompt_brand = (brand + " ") if brand else ""
    prompt = (
        f"You are a polite brand social media manager for {prompt_brand}. "
        f"A user wrote: \"{mention_text}\". Write a short, professional, empathetic public reply (1-3 sentences) "
        f"that acknowledges the concern, offers help/next steps, and invites the user to DM or a support link if needed."
    )

    # Only call OpenAI if imported and api key present
    if OPENAI_API_KEY and OPENAI_AVAILABLE and openai is not None:
        try:
            # Some openai SDK versions expose ChatCompletion; check first
            if hasattr(openai, "ChatCompletion"):
                resp = openai.ChatCompletion.create(
                    model=OPENAI_DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": "You write concise, helpful public support responses for brand social media."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=120,
                    temperature=0.2,
                )
                # Extract answer in the usual shape
                if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                    ai_text = resp["choices"][0].get("message", {}).get("content")
                    if ai_text:
                        return ai_text.strip()
            # If ChatCompletion not available or response shape unexpected, try the newer chat API if present:
            if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
                # This path may or may not exist depending on openai version
                resp2 = openai.chat.completions.create(
                    model=OPENAI_DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": "You write concise, helpful public support responses for brand social media."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=120,
                    temperature=0.2,
                )
                # resp2 structure may differ; try to extract text robustly
                try:
                    choices = getattr(resp2, "choices", None) or (resp2.get("choices") if isinstance(resp2, dict) else None)
                    if choices:
                        first = choices[0]
                        # multiple possible fields: 'message'->'content' or 'text'
                        if isinstance(first, dict):
                            ai_text = (first.get("message") or {}).get("content") or first.get("text")
                        else:
                            ai_text = None
                        if ai_text:
                            return ai_text.strip()
                except Exception:
                    pass
        except Exception as e:
            logger.exception("OpenAI call failed, falling back to template. Error: %s", e)

    # Fallback templated reply
    snippet = (mention_text[:120] + "...") if len(mention_text) > 120 else mention_text
    return (f"Thanks for flagging this. We're sorry to hear about your experience — we'd like to help. "
            f"Can you DM us with details or reach out at support@example.com? (Ref: \"{snippet}\")")


# ---------------------------
# API endpoints
# ---------------------------
@app.post("/process_batch")
async def process_batch(req: ProcessRequest):
    try:
        mentions = req.mentions or []
        if not mentions:
            return {"status": "ok", "message": "no mentions provided", "data": {}}

        mention_rows = []
        texts: List[str] = []
        for m in mentions:
            text = (m.text or "").strip()
            if not text:
                continue
            sent = analyze_sentiment(text)
            mention_rows.append({
                "id": m.id,
                "platform": m.platform,
                "author": m.author,
                "created_at": m.created_at,
                "text": text,
                "compound": sent["compound"],
                "label": sent["label"],
                "scores": sent["scores"]
            })
            texts.append(text)

        df = pd.DataFrame(mention_rows)
        if df.empty:
            raise HTTPException(status_code=400, detail="All mentions were empty after normalization.")

        positive_count = int((df["label"] == "positive").sum())
        negative_count = int((df["label"] == "negative").sum())
        neutral_count = int((df["label"] == "neutral").sum())
        total = positive_count + neutral_count + negative_count

        reputation = compute_reputation_score(positive_count, neutral_count, negative_count)
        keywords = extract_trending_keywords(texts, top_k=TOP_K_KEYWORDS)
        alerts = detect_critical_alerts(
            positive_count, neutral_count, negative_count,
            req.historical_negative_ratio, req.historical_window_size
        )

        negative_mentions = df[df["label"] == "negative"].to_dict(orient="records")
        suggested_responses = []
        for nm in negative_mentions[:10]:
            ai_draft = await generate_response_draft_openai(req.brand, nm["text"], nm.get("author"))
            suggested_responses.append({
                "mention_id": nm.get("id"),
                "platform": nm.get("platform"),
                "text_snippet": (nm["text"][:200] + "...") if len(nm["text"]) > 200 else nm["text"],
                "draft": ai_draft
            })

        return {
            "status": "ok",
            "data": {
                "summary": {
                    "total_mentions": total,
                    "positive": positive_count,
                    "neutral": neutral_count,
                    "negative": negative_count,
                },
                "reputation_score": reputation,
                "trending_keywords": keywords,
                "alerts": alerts,
                "suggested_responses": suggested_responses
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "component": "sentiment_agent"}


# ---------------------------
# Local test + run
# ---------------------------
if __name__ == "__main__":
    import uvicorn, asyncio, json

    sample = {
        "brand": "TestBrand",
        "mentions": [
            {"id": "m1", "platform": "twitter", "text": "Love the new update — great work!"},
            {"id": "m2", "platform": "reddit", "text": "Delayed delivery, very upset"},
            {"id": "m3", "platform": "twitter", "text": "Product is okay, nothing special."},
            {"id": "m4", "platform": "twitter", "text": "Really bad experience, broken on arrival."},
            {"id": "m5", "platform": "reddit", "text": "Amazing battery life!"}
        ]
    }

    async def local_test():
        req = ProcessRequest(**sample)
        res = await process_batch(req)
        print(json.dumps(res, indent=2))

    print("Starting Sentiment Agent on http://127.0.0.1:8001 ...")
    asyncio.run(local_test())
    uvicorn.run("sentiment_agent:app", host="0.0.0.0", port=8001, log_level="info")
