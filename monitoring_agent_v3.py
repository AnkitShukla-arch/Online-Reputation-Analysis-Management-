import os
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import time
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional modules
try:
    import snscrape.modules.twitter as sntwitter
except Exception:
    sntwitter = None

try:
    import praw
except Exception:
    praw = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
analyzer = SentimentIntensityAnalyzer()

# Convert date safely
def safe_date_str(value):
    try:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, time.struct_time):
            return datetime(*value[:6]).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return value
    except Exception:
        return ""
    return ""

# ---------- Google News ----------
def scrape_google_news(brand, keywords, days, limit):
    results = []
    try:
        q = quote_plus(f"{brand} {' '.join(keywords)}")
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        rss_url = f"https://news.google.com/rss/search?q={q}+after:{since}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(rss_url, timeout=10)
        resp.raise_for_status()

        xml_text = resp.text.replace("xmlns=", "ns=")
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item")[:limit]:
            results.append({
                "platform": "news",
                "author": (item.findtext("source") or "Unknown").strip(),
                "text": (item.findtext("title") or "") + " " + (item.findtext("description") or ""),
                "date": safe_date_str(item.findtext("pubDate")),
                "url": item.findtext("link") or "",
                "engagement": 0,
                "brand": brand,
                "collected_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            })
    except Exception as e:
        print("[News] Error:", e)
    return results

# ---------- Twitter (robust) ----------
def scrape_twitter(brand, keywords, days, limit):
    results = []
    if sntwitter is None:
        print("[Twitter] snscrape not available.")
        return results

    try:
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        # build query safely (avoid extra spaces when keywords empty)
        kw_part = " ".join(k.strip() for k in keywords if k and k.strip())
        if kw_part:
            query = f'"{brand}" {kw_part} since:{since}'
        else:
            query = f'"{brand}" since:{since}'

        print(f"[Twitter] Query: {query}")
        scraper = sntwitter.TwitterSearchScraper(query)
        for i, tweet in enumerate(scraper.get_items()):
            if i >= limit:
                break

            # author
            author = ""
            try:
                author = getattr(getattr(tweet, "user", None), "username", "") or getattr(getattr(tweet, "user", None), "displayname", "") or ""
            except Exception:
                author = ""

            # text/content
            content = ""
            try:
                content = getattr(tweet, "content", "") or getattr(tweet, "renderedContent", "") or ""
            except Exception:
                content = ""

            # date
            date_str = ""
            try:
                date_str = safe_date_str(getattr(tweet, "date", None))
            except Exception:
                date_str = ""

            # url / tweet id
            try:
                tweet_id = getattr(tweet, "id", None) or getattr(tweet, "tweetId", None)
                if tweet_id and author:
                    url = f"https://twitter.com/{author}/status/{tweet_id}"
                else:
                    url = getattr(tweet, "url", "") or ""
            except Exception:
                url = ""

            # engagement counts (robustly handle missing fields / different names)
            try:
                like_count = int(getattr(tweet, "likeCount", None) or getattr(tweet, "likes", 0) or 0)
            except Exception:
                like_count = 0
            try:
                retweet_count = int(getattr(tweet, "retweetCount", None) or getattr(tweet, "retweets", 0) or 0)
            except Exception:
                retweet_count = 0
            try:
                reply_count = int(getattr(tweet, "replyCount", None) or getattr(tweet, "replies", 0) or 0)
            except Exception:
                reply_count = 0

            engagement = like_count + retweet_count + reply_count

            # skip empty content
            if not content:
                continue

            results.append({
                "platform": "twitter",
                "author": author or "unknown",
                "text": content[:400],
                "date": date_str or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "url": url,
                "engagement": engagement,
                "brand": brand,
                "collected_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            })

    except Exception as e:
        print("[Twitter] Error:", e)

    return results


# ---------- Reddit ----------
def scrape_reddit(brand, keywords, limit):
    results = []
    if not praw or not os.getenv("REDDIT_CLIENT_ID"):
        print("[Reddit] Skipping — missing credentials.")
        return results

    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "monitoring-agent")
        )
        query = f"{brand} {' OR '.join(keywords)}"
        for post in reddit.subreddit("all").search(query, limit=limit):
            results.append({
                "platform": "reddit",
                "author": str(post.author),
                "text": (post.title or "") + " " + (post.selftext[:300] or ""),
                "date": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "url": f"https://www.reddit.com{post.permalink}",
                "engagement": int(post.score),
                "brand": brand,
                "collected_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            })
    except Exception as e:
        print("[Reddit] Error:", e)
    return results

# ---------- Orchestrator ----------
def run_monitor(brand, platforms, keywords, days=7, limit_per_platform=10):
    all_items = []
    if "news" in platforms:
        all_items += scrape_google_news(brand, keywords, days, limit_per_platform)
    if "twitter" in platforms:
        all_items += scrape_twitter(brand, keywords, days, limit_per_platform)
    if "reddit" in platforms:
        all_items += scrape_reddit(brand, keywords, limit_per_platform)

    if not all_items:
        print("[Monitor] No data found.")
        return None

    # Sentiment
    for item in all_items:
        s = analyzer.polarity_scores(item["text"])
        item["sentiment_score"] = s["compound"]
        item["sentiment"] = "positive" if s["compound"] > 0.05 else "negative" if s["compound"] < -0.05 else "neutral"

    df = pd.DataFrame(all_items)
    filename = os.path.join(OUTPUT_DIR, f"monitoring_{brand}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"[Monitor] Saved {len(df)} rows to {filename}")
    return filename

if __name__ == "__main__":
    run_monitor("Tesla", ["news", "twitter"], ["autopilot", "battery"], 7, 10)
