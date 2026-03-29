"""
ET Nexus — The AI-Native Newsroom
FastAPI Backend with mock data and simulated AI endpoints.
"""

import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from typing import Optional
from urllib.parse import urljoin

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="ET Nexus API",
    description="Backend for ET Nexus — The AI-Native Newsroom",
    version="0.1.0",
)

_cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
CORS_ALLOW_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class BriefingRequest(BaseModel):
    article_id: str

class ChatRequest(BaseModel):
    question: str
    context_id: str

class TranslateRequest(BaseModel):
    text: str
    target_language: str

class BatchTranslateRequest(BaseModel):
    texts: list[str]
    target_language: str

class VideoRequest(BaseModel):
    article_id: str

class StoryArcRequest(BaseModel):
    topic: str

class NavigatorRequest(BaseModel):
    topic: str
    persona: str = "investor"

# ─────────────────────────────────────────────
# Live Article Ingestion
# ─────────────────────────────────────────────

ARTICLES_DB = {}
INGEST_LOCK = asyncio.Lock()
LAST_INGESTED_AT: Optional[datetime] = None
FEED_REFRESH_TTL_SECONDS = int(os.getenv("FEED_REFRESH_TTL_SECONDS", "900"))
MAX_FEED_ARTICLES = int(os.getenv("MAX_FEED_ARTICLES", "12"))
MAX_STORY_ARC_SUMMARIES = int(os.getenv("MAX_STORY_ARC_SUMMARIES", "6"))
MAX_RELATED_ARTICLES = int(os.getenv("MAX_RELATED_ARTICLES", "4"))
INGEST_WITH_GROQ = os.getenv("INGEST_WITH_GROQ", "false").lower() == "true"
TRANSLATION_CONCURRENCY = int(os.getenv("TRANSLATION_CONCURRENCY", "2"))
ET_RSS_URLS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/2146842.cms",
    "https://economictimes.indiatimes.com/tech/rssfeeds/13352306.cms",
    "https://b2b.economictimes.indiatimes.com/rss/startup",
    "https://education.economictimes.indiatimes.com/rss/topstories",
]
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
_STOP_WORDS = {
    "about", "after", "again", "against", "almost", "among", "because", "been", "being",
    "business", "company", "could", "first", "from", "have", "into", "market", "markets",
    "more", "news", "over", "said", "says", "that", "their", "there", "these", "they",
    "this", "today", "under", "what", "when", "where", "which", "will", "with", "would",
}
_SENTIMENT_SCORE = {
    "positive": 0.65,
    "negative": -0.65,
    "neutral": 0.0,
}
_TRANSLATION_CACHE: dict[tuple[str, str], dict] = {}
_SUPPORTED_TRANSLATION_LANGUAGES = {"English", "Hindi", "Tamil", "Telugu", "Bengali"}
_TRANSLATION_SEMAPHORE = asyncio.Semaphore(TRANSLATION_CONCURRENCY)
_LANGUAGE_SCRIPT_RANGES = {
    "Hindi": ("\u0900", "\u097F"),
    "Tamil": ("\u0B80", "\u0BFF"),
    "Telugu": ("\u0C00", "\u0C7F"),
    "Bengali": ("\u0980", "\u09FF"),
}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    article_div = soup.find("div", class_="artText")
    if article_div:
        return _clean_text(article_div.get_text(" ", strip=True))

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    combined = " ".join([p for p in paragraphs if p])
    return _clean_text(combined)


def _extract_article_image(html: str, article_url: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")

    meta_candidates = [
        {"property": "og:image"},
        {"name": "og:image"},
        {"name": "twitter:image"},
        {"property": "twitter:image"},
    ]
    for attrs in meta_candidates:
        tag = soup.find("meta", attrs=attrs)
        content = tag.get("content") if tag else None
        if content:
            return urljoin(article_url, content)

    link_tag = soup.find("link", rel="image_src")
    if link_tag and link_tag.get("href"):
        return urljoin(article_url, link_tag["href"])

    for img in soup.find_all("img", src=True):
        src = img.get("src", "")
        if not src:
            continue
        lowered = src.lower()
        if any(token in lowered for token in ("photo", "img", "indiatimes", "economictimes")):
            return urljoin(article_url, src)

    return ""


def _entry_date(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime(*parsed[:6]).strftime("%Y-%m-%d")
    return datetime.utcnow().strftime("%Y-%m-%d")


def _entry_tags(entry) -> list[str]:
    tags = []
    for tag in entry.get("tags", []):
        term = getattr(tag, "term", None)
        if term:
            tags.append(str(term))
    return tags


def _stable_article_id(link: str) -> str:
    return hashlib.sha1(link.encode("utf-8")).hexdigest()[:16]


def _entry_sort_key(entry) -> tuple:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return tuple(parsed[:6])
    return ()


def _infer_persona_from_text(text: str, title: str = "") -> str:
    combined = f"{title} {text}".lower()

    keyword_groups = {
        "founder": [
            "startup", "founder", "funding", "venture", "vc", "seed", "series a",
            "series b", "valuation", "burn rate", "product", "saas", "entrepreneur",
        ],
        "student": [
            "explainer", "what is", "guide", "basics", "learn", "education", "curriculum",
            "beginner", "framework", "concept", "how to",
        ],
        "investor": [
            "stock", "market", "sensex", "nifty", "earnings", "shares", "bond", "ipo",
            "valuation", "portfolio", "inflation", "interest rate", "fed", "rbi",
        ],
    }

    scores = {
        persona: sum(combined.count(keyword) for keyword in keywords)
        for persona, keywords in keyword_groups.items()
    }
    best_persona = max(scores, key=scores.get)
    return best_persona if scores[best_persona] > 0 else "investor"


def _fallback_article_meta(text: str, title: str = "") -> dict:
    two_sentence_summary = _clean_text(text)[:320]
    persona = _infer_persona_from_text(text, title)
    return {
        "persona": persona,
        "sentiment": "neutral",
        "summary": two_sentence_summary or "Market update from Economic Times.",
    }


async def _classify_article_with_groq(full_text: str, title: str) -> dict:
    if not groq_client:
        return _fallback_article_meta(full_text, title)

    system_prompt = (
        "You are an economic news analyst. Return ONLY raw JSON with exact keys: "
        "persona, sentiment, summary. "
        "Use these strict persona definitions: "
        "investor: Focuses on stock markets, funding rounds, acquisitions, and macroeconomics. "
        "founder: Focuses on startup growth, leadership, venture capital, and scaling businesses. "
        "student: Focuses on artificial intelligence, machine learning, web development frameworks, tech upskilling, internships, and education policies. "
        "persona must be one of investor|founder|student. "
        "sentiment must be one of bullish|bearish|neutral. "
        "summary must be one concise 2-sentence string. "
        "No markdown, no extra text, no extra keys."
    )

    user_prompt = (
        f"Title: {title}\n"
        f"Article text:\n{full_text[:12000]}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = _extract_json_object(completion.choices[0].message.content or "")

        persona = str(parsed.get("persona", "investor")).lower()
        if persona not in {"investor", "founder", "student"}:
            persona = "investor"

        sentiment = str(parsed.get("sentiment", "neutral")).lower()
        if sentiment not in {"bullish", "bearish", "neutral"}:
            sentiment = "neutral"

        summary = _clean_text(str(parsed.get("summary", "")))
        if not summary:
            summary = _fallback_article_meta(full_text, title)["summary"]

        return {
            "persona": persona,
            "sentiment": sentiment,
            "summary": summary,
        }
    except Exception:
        return _fallback_article_meta(full_text, title)


async def _classify_article_for_ingest(full_text: str, title: str) -> dict:
    if INGEST_WITH_GROQ:
        return await _classify_article_with_groq(full_text, title)
    return _fallback_article_meta(full_text, title)


def _articles_are_stale() -> bool:
    if not ARTICLES_DB:
        return True
    if LAST_INGESTED_AT is None:
        return True
    return datetime.utcnow() - LAST_INGESTED_AT >= timedelta(seconds=FEED_REFRESH_TTL_SECONDS)


def _article_datetime(article: dict) -> datetime:
    date_value = article.get("date")
    if not date_value:
        return datetime.min
    try:
        return datetime.strptime(date_value, "%Y-%m-%d")
    except ValueError:
        return datetime.min


async def _ensure_articles_loaded(force_refresh: bool = False) -> None:
    if not force_refresh and not _articles_are_stale():
        return

    async with INGEST_LOCK:
        if not force_refresh and not _articles_are_stale():
            return
        await ingest_live_news()


async def _fetch_full_article_text(link: str, fallback_text: str = "") -> str:
    if not link:
        return _clean_text(fallback_text)

    try:
        response = await asyncio.to_thread(requests.get, link, headers=_REQUEST_HEADERS, timeout=15)
        response.raise_for_status()
        extracted = _extract_article_text(response.text)
        if extracted:
            return extracted
    except Exception:
        pass

    return _clean_text(fallback_text)


async def _fetch_article_assets(link: str, fallback_text: str = "") -> tuple[str, str]:
    if not link:
        return _clean_text(fallback_text), ""

    try:
        response = await asyncio.to_thread(requests.get, link, headers=_REQUEST_HEADERS, timeout=15)
        response.raise_for_status()
        html = response.text
        text = _extract_article_text(html) or _clean_text(fallback_text)
        image_url = _extract_article_image(html, link)
        return text, image_url
    except Exception:
        return _clean_text(fallback_text), ""


async def _ingest_rss_news() -> dict[str, dict]:
    sentiment_map = {
        "bullish": "positive",
        "bearish": "negative",
        "neutral": "neutral",
    }

    entries = []
    for rss_url in ET_RSS_URLS:
        feed = await asyncio.to_thread(feedparser.parse, rss_url)
        if hasattr(feed, "entries"):
            entries.extend(feed.entries)

    entries.sort(key=_entry_sort_key, reverse=True)

    valid_entries = []
    seen_links = set()
    for entry in entries:
        title = _clean_text(entry.get("title", ""))
        description = _clean_text(BeautifulSoup(entry.get("description", ""), "html.parser").get_text(" ", strip=True))
        link = entry.get("link", "")
        if not title:
            continue
        if "live updates" in title.lower():
            continue
        if not description:
            continue
        if not link or link in seen_links:
            continue
        seen_links.add(link)
        valid_entries.append(entry)
        if len(valid_entries) >= MAX_FEED_ARTICLES:
            break

    ingested = {}
    for entry in valid_entries:
        link = entry.get("link", "")
        if not link:
            continue

        full_text, image_url = await _fetch_article_assets(link, fallback_text=entry.get("description", ""))
        if not full_text:
            continue

        title = _clean_text(entry.get("title", "Economic Times Market Update"))
        llm_meta = await _classify_article_for_ingest(full_text, title)

        article_id = _stable_article_id(link)
        read_time = max(1, len(full_text.split()) // 220)
        persona = llm_meta["persona"]
        market_sentiment = llm_meta["sentiment"]

        article = {
            "id": article_id,
            "title": title,
            "summary": llm_meta["summary"],
            "content": full_text,
            "full_text": full_text,
            "author": _clean_text(entry.get("author", "Economic Times")) or "Economic Times",
            "date": _entry_date(entry),
            "category": "Markets",
            "tags": _entry_tags(entry) or ["markets"],
            "sentiment": sentiment_map.get(market_sentiment, "neutral"),
            "market_sentiment": market_sentiment,
            "image_url": image_url,
            "persona_relevance": [persona],
            "source": "Economic Times",
            "read_time": read_time,
            "link": link,
        }
        ingested[article_id] = article

    return ingested


async def ingest_live_news() -> dict:
    global LAST_INGESTED_AT

    ingested = await _ingest_rss_news()

    if ingested:
        ARTICLES_DB.clear()
        ARTICLES_DB.update(ingested)
    LAST_INGESTED_AT = datetime.utcnow()

    return {
        "ingested_count": len(ARTICLES_DB),
        "article_ids": list(ARTICLES_DB.keys()),
    }


def _extract_json_object(raw_text: str) -> dict:
    """Extract a JSON object from model output, even if wrapped in markdown fences."""
    if not raw_text:
        raise ValueError("Empty model response")

    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end + 1])


def _tokenize_text(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    filtered = [token for token in tokens if token not in _STOP_WORDS]
    return Counter(filtered)


def _article_topic_counter(article: dict) -> Counter[str]:
    combined = " ".join([
        article.get("title", ""),
        article.get("summary", ""),
        " ".join(article.get("tags", [])),
        (article.get("full_text") or article.get("content", ""))[:1200],
    ])
    return _tokenize_text(combined)


def _extract_key_players_from_text(text: str) -> list[str]:
    matches = re.findall(r"\b[A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,3}\b", text or "")
    candidates = []
    for match in matches:
        cleaned = _clean_text(match)
        if len(cleaned) < 3:
            continue
        if cleaned.lower() in {"economic times", "et nexus"}:
            continue
        candidates.append(cleaned)
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped[:8]


def _article_similarity_score(seed_article: dict, candidate_article: dict) -> float:
    if seed_article["id"] == candidate_article["id"]:
        return -1.0

    seed_counter = _article_topic_counter(seed_article)
    candidate_counter = _article_topic_counter(candidate_article)
    shared_terms = set(seed_counter) & set(candidate_counter)
    shared_term_score = sum(min(seed_counter[term], candidate_counter[term]) for term in shared_terms)

    shared_tags = len(set(seed_article.get("tags", [])) & set(candidate_article.get("tags", [])))
    shared_persona = len(set(seed_article.get("persona_relevance", [])) & set(candidate_article.get("persona_relevance", [])))
    same_category = 1 if seed_article.get("category") == candidate_article.get("category") else 0
    same_sentiment = 1 if seed_article.get("sentiment") == candidate_article.get("sentiment") else 0

    return (shared_term_score * 1.5) + (shared_tags * 2.0) + (shared_persona * 1.5) + same_category + (same_sentiment * 0.5)


def _filter_articles_for_persona(persona: str) -> list[dict]:
    persona_normalized = persona.lower()
    articles = list(ARTICLES_DB.values())
    if persona_normalized == "general":
        return articles
    return [
        article for article in articles
        if persona_normalized in article.get("persona_relevance", [])
    ]


def _get_related_articles(seed_article: dict, persona: str = "general", limit: int = MAX_RELATED_ARTICLES) -> list[dict]:
    candidates = []
    for article in _filter_articles_for_persona(persona):
        score = _article_similarity_score(seed_article, article)
        if score <= 0:
            continue
        candidates.append((score, article))

    candidates.sort(key=lambda item: (item[0], _article_datetime(item[1])), reverse=True)
    return [article for _, article in candidates[:limit]]


def _build_key_players(seed_article: dict, related_articles: list[dict]) -> list[dict]:
    sources = [seed_article, *related_articles]
    player_counts: Counter[str] = Counter()
    player_sentiments: dict[str, list[str]] = {}

    for article in sources:
        text = " ".join([article.get("title", ""), article.get("summary", "")])
        for player in _extract_key_players_from_text(text):
            player_counts[player] += 1
            player_sentiments.setdefault(player, []).append(article.get("sentiment", "neutral"))

    key_players = []
    for player, _count in player_counts.most_common(5):
        sentiments = player_sentiments.get(player, ["neutral"])
        dominant_sentiment = max(set(sentiments), key=sentiments.count)
        role = "Central figure in the related coverage cluster."
        if player.lower() in seed_article.get("title", "").lower():
            role = "Directly referenced in the lead article."
        key_players.append({
            "name": player,
            "role": role,
            "sentiment": dominant_sentiment,
        })

    if key_players:
        return key_players

    return [{
        "name": seed_article.get("source", "Economic Times"),
        "role": "Primary source covering the underlying development.",
        "sentiment": seed_article.get("sentiment", "neutral"),
    }]


def _story_arc_slug(seed_article: dict) -> str:
    return seed_article["id"]


def _derive_story_arc_title(seed_article: dict, related_articles: list[dict]) -> str:
    lead_title = _clean_text(seed_article.get("title", "Story Arc"))
    lead_title = re.split(r"\s[:;|-]\s|[:;|-]", lead_title, maxsplit=1)[0].strip() or lead_title
    if len(lead_title) > 72:
        lead_title = f"{lead_title[:69].rstrip()}..."
    shared_tags = []
    for article in [seed_article, *related_articles]:
        shared_tags.extend(article.get("tags", []))
    common_tag = Counter(shared_tags).most_common(1)
    if common_tag:
        topic = common_tag[0][0].replace("-", " ").title()
        if topic.lower() not in lead_title.lower():
            return f"{lead_title} | {topic}"
    return lead_title


def _derive_story_arc_description(seed_article: dict, related_articles: list[dict]) -> str:
    cluster_size = 1 + len(related_articles)
    return (
        f"Tracing how {seed_article.get('title', 'this story').lower()} connects to "
        f"{cluster_size - 1} related developments across the live feed."
    )


def _build_story_arc_fallback(seed_article: dict, related_articles: list[dict]) -> dict:
    cluster = [seed_article, *related_articles]
    cluster.sort(key=_article_datetime)

    events = []
    trajectory = []
    for index, article in enumerate(cluster, start=1):
        impact_score = round(min(9.5, 5.0 + (_article_similarity_score(seed_article, article) if article["id"] != seed_article["id"] else 3.5)), 1)
        sentiment = article.get("sentiment", "neutral")
        events.append({
            "date": article.get("date", datetime.utcnow().strftime("%Y-%m-%d")),
            "title": article.get("title", "Related development"),
            "description": article.get("summary", "No summary available."),
            "sentiment": sentiment,
            "impact_score": impact_score,
            "related_articles": [article["id"]],
        })
        trajectory.append({
            "date": article.get("date", datetime.utcnow().strftime("%Y-%m-%d")),
            "score": _SENTIMENT_SCORE.get(sentiment, 0.0),
        })

    tags = [tag for article in cluster for tag in article.get("tags", [])]
    common_tags = [tag for tag, _ in Counter(tags).most_common(3)]
    common_tags_text = ", ".join(common_tags) if common_tags else "market structure and sector response"
    predictions = [
        f"Expect follow-on coverage around {common_tags_text}.",
        "Watch whether the lead narrative broadens into a cross-sector trend over the next refresh cycle.",
        "Monitor whether sentiment improves or weakens as related articles continue to accumulate.",
    ]

    return {
        "id": _story_arc_slug(seed_article),
        "title": _derive_story_arc_title(seed_article, related_articles),
        "description": _derive_story_arc_description(seed_article, related_articles),
        "status": "active",
        "events": events,
        "key_players": _build_key_players(seed_article, related_articles),
        "sentiment_trajectory": trajectory,
        "predictions": predictions,
    }


async def _build_story_arc_with_groq(seed_article: dict, related_articles: list[dict]) -> dict:
    fallback_arc = _build_story_arc_fallback(seed_article, related_articles)
    if not groq_client:
        return fallback_arc

    cluster = [seed_article, *related_articles]
    cluster.sort(key=_article_datetime)
    article_context = "\n\n".join([
        (
            f"Article ID: {article['id']}\n"
            f"Date: {article.get('date', '')}\n"
            f"Title: {article.get('title', '')}\n"
            f"Summary: {article.get('summary', '')}\n"
            f"Tags: {', '.join(article.get('tags', []))}\n"
            f"Sentiment: {article.get('sentiment', 'neutral')}"
        )
        for article in cluster
    ])

    system_prompt = (
        "You are an economic newsroom analyst creating article-dependent story arcs from a retrieved article set. "
        "Return ONLY raw JSON with exact keys: id, title, description, status, events, key_players, sentiment_trajectory, predictions. "
        "status must be active. "
        "events must be 3-5 objects with keys date, title, description, sentiment, impact_score, related_articles. "
        "sentiment must be one of positive|negative|neutral. impact_score must be 0-10. related_articles must only contain provided article IDs. "
        "key_players must be 2-5 objects with keys name, role, sentiment. "
        "sentiment_trajectory must be 3-5 objects with keys date and score where score is between -1 and 1. "
        "predictions must be 2-4 concise strings. No markdown, no extra text."
    )
    user_prompt = (
        f"Lead article ID: {seed_article['id']}\n"
        f"Lead article title: {seed_article.get('title', '')}\n"
        f"Retrieved article set:\n{article_context}"
    )

    valid_article_ids = {article["id"] for article in cluster}

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = _extract_json_object(completion.choices[0].message.content or "")

        events = []
        for raw_event in parsed.get("events", []):
            if not isinstance(raw_event, dict):
                continue
            related_ids = [
                str(article_id) for article_id in raw_event.get("related_articles", [])
                if str(article_id) in valid_article_ids
            ]
            events.append({
                "date": str(raw_event.get("date", seed_article.get("date", datetime.utcnow().strftime("%Y-%m-%d")))),
                "title": _clean_text(str(raw_event.get("title", ""))) or fallback_arc["events"][0]["title"],
                "description": _clean_text(str(raw_event.get("description", ""))) or fallback_arc["events"][0]["description"],
                "sentiment": str(raw_event.get("sentiment", "neutral")).lower() if str(raw_event.get("sentiment", "neutral")).lower() in {"positive", "negative", "neutral"} else "neutral",
                "impact_score": round(max(0.0, min(10.0, float(raw_event.get("impact_score", 5.0)))), 1),
                "related_articles": related_ids or [seed_article["id"]],
            })

        key_players = []
        for raw_player in parsed.get("key_players", []):
            if not isinstance(raw_player, dict):
                continue
            sentiment = str(raw_player.get("sentiment", "neutral")).lower()
            key_players.append({
                "name": _clean_text(str(raw_player.get("name", ""))),
                "role": _clean_text(str(raw_player.get("role", ""))) or "Relevant stakeholder in the coverage cluster.",
                "sentiment": sentiment if sentiment in {"positive", "negative", "neutral"} else "neutral",
            })
        key_players = [player for player in key_players if player["name"]][:5]

        sentiment_trajectory = []
        for raw_point in parsed.get("sentiment_trajectory", []):
            if not isinstance(raw_point, dict):
                continue
            try:
                score = round(max(-1.0, min(1.0, float(raw_point.get("score", 0.0)))), 2)
            except (TypeError, ValueError):
                score = 0.0
            sentiment_trajectory.append({
                "date": str(raw_point.get("date", seed_article.get("date", datetime.utcnow().strftime("%Y-%m-%d")))),
                "score": score,
            })

        predictions = [
            _clean_text(str(prediction))
            for prediction in parsed.get("predictions", [])
            if _clean_text(str(prediction))
        ][:4]

        return {
            "id": _story_arc_slug(seed_article),
            "title": _clean_text(str(parsed.get("title", ""))) or fallback_arc["title"],
            "description": _clean_text(str(parsed.get("description", ""))) or fallback_arc["description"],
            "status": "active",
            "events": events or fallback_arc["events"],
            "key_players": key_players or fallback_arc["key_players"],
            "sentiment_trajectory": sentiment_trajectory or fallback_arc["sentiment_trajectory"],
            "predictions": predictions or fallback_arc["predictions"],
        }
    except Exception:
        return fallback_arc


def _build_story_arc_summary(seed_article: dict, related_articles: list[dict]) -> dict:
    return {
        "id": _story_arc_slug(seed_article),
        "title": _derive_story_arc_title(seed_article, related_articles),
        "description": _derive_story_arc_description(seed_article, related_articles),
        "status": "active",
        "event_count": 1 + len(related_articles),
        "player_count": len(_build_key_players(seed_article, related_articles)),
    }


def _default_briefing(article: dict) -> dict:
    return {
        "bullets": [
            article.get("summary", "Key developments are covered in this article."),
            f"Category focus: {article.get('category', 'General')}.",
            f"Top themes: {', '.join(article.get('tags', [])[:3]) or 'business, markets, analysis'}.",
            "The story includes market impact, stakeholder implications, and forward-looking signals.",
        ],
        "sentiment": article.get("sentiment", "neutral"),
        "confidence_score": 70,
    }


def _default_chat_response(article: dict, question: str) -> dict:
    context_title = article.get("title", "the selected article")
    return {
        "response": (
            f"Based on \"{context_title}\", a concise takeaway is: "
            f"{article.get('summary', 'the article highlights key business developments and their implications')}. "
            f"Your question was: \"{question}\"."
        ),
        "sources": [context_title],
    }


async def generate_briefing(article: dict) -> dict:
    """Generate article briefing with Groq and a strict JSON output contract."""
    if not groq_client:
        return _default_briefing(article)

    system_prompt = (
        "You are a business news analyst. "
        "Return ONLY raw JSON with this exact schema and no extra keys: "
        "{\"bullets\": [string], \"sentiment\": \"positive|negative|neutral\", \"confidence_score\": integer}. "
        "Rules: bullets must be 4-6 concise factual points. confidence_score must be 0-100 integer. "
        "No markdown fences. No additional text."
    )
    user_prompt = (
        f"Article title: {article.get('title', '')}\n"
        f"Article summary: {article.get('summary', '')}\n"
        f"Article content:\n{article.get('full_text') or article.get('content', '')}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content or ""
        parsed = _extract_json_object(content)

        bullets = parsed.get("bullets", [])
        if not isinstance(bullets, list):
            raise ValueError("Invalid bullets format")
        bullets = [str(b).strip() for b in bullets if str(b).strip()]
        if not bullets:
            raise ValueError("No bullets returned")

        sentiment = str(parsed.get("sentiment", "neutral")).lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"

        confidence_score = int(parsed.get("confidence_score", 70))
        confidence_score = max(0, min(100, confidence_score))

        return {
            "bullets": bullets,
            "sentiment": sentiment,
            "confidence_score": confidence_score,
        }
    except Exception:
        return _default_briefing(article)


async def generate_chat_response(question: str, context_id: str) -> dict:
    """Answer a user question grounded strictly in the selected article text."""
    article = ARTICLES_DB.get(context_id)
    if not article:
        return {
            "response": "I could not find context for this article. Please open the article again and retry.",
            "sources": ["unknown"],
        }

    if not groq_client:
        return _default_chat_response(article, question)

    context_title = article["title"]
    system_prompt = (
        "You are a financial news assistant. "
        "Answer the user question using strictly and only the provided article text. "
        "If the answer is not explicitly supported by the text, say that the article does not provide that detail. "
        "Keep the answer concise and factual."
    )
    user_prompt = (
        f"Article title: {context_title}\n"
        f"Article text:\n{article.get('full_text') or article.get('content', '')}\n\n"
        f"User question: {question}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response_text = (completion.choices[0].message.content or "").strip()
        if not response_text:
            raise ValueError("Empty chat completion")

        return {
            "response": response_text,
            "sources": [context_title],
        }
    except Exception:
        return _default_chat_response(article, question)


async def generate_translation(text: str, target_language: str) -> dict:
    cleaned_text = (text or "").strip()
    target_language = target_language.strip()

    if not cleaned_text:
        return {
            "original": text,
            "translated": "",
            "target_language": target_language,
            "culturally_adapted": False,
            "note": "Empty text received.",
        }

    if target_language not in _SUPPORTED_TRANSLATION_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language '{target_language}'. Choose from: {sorted(_SUPPORTED_TRANSLATION_LANGUAGES)}",
        )

    if target_language == "English":
        return {
            "original": text,
            "translated": cleaned_text,
            "target_language": target_language,
            "culturally_adapted": False,
            "note": "English selected; original text returned.",
        }

    cache_key = (target_language, hashlib.sha1(cleaned_text.encode("utf-8")).hexdigest())
    cached = _TRANSLATION_CACHE.get(cache_key)
    if cached:
        return cached

    def _normalize_for_compare(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip().lower()

    def _contains_target_script(value: str, language: str) -> bool:
        script_range = _LANGUAGE_SCRIPT_RANGES.get(language)
        if not script_range:
            return True
        start, end = script_range
        return any(start <= char <= end for char in value)

    def _looks_translated(source_text: str, translated_text: str, language: str) -> bool:
        source_norm = _normalize_for_compare(source_text)
        translated_norm = _normalize_for_compare(translated_text)

        if not translated_norm:
            return False

        if translated_norm == source_norm and len(source_norm) > 24:
            return False

        if language in _LANGUAGE_SCRIPT_RANGES and len(source_norm) > 24:
            return _contains_target_script(translated_text, language)

        return True

    def _chunk_text_for_translation(source_text: str, max_chars: int = 2400) -> list[str]:
        paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", source_text) if segment.strip()]
        if not paragraphs:
            paragraphs = [source_text]

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) <= max_chars:
                candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = paragraph
                continue

            sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", paragraph) if segment.strip()]
            for sentence in sentences:
                if len(sentence) > max_chars:
                    words = sentence.split()
                    sentence_piece = ""
                    for word in words:
                        candidate = f"{sentence_piece} {word}".strip()
                        if len(candidate) <= max_chars:
                            sentence_piece = candidate
                        else:
                            if sentence_piece:
                                candidate_chunk = f"{current}\n\n{sentence_piece}".strip() if current else sentence_piece
                                if len(candidate_chunk) <= max_chars:
                                    current = candidate_chunk
                                else:
                                    if current:
                                        chunks.append(current)
                                    current = sentence_piece
                            sentence_piece = word
                    if sentence_piece:
                        candidate_chunk = f"{current}\n\n{sentence_piece}".strip() if current else sentence_piece
                        if len(candidate_chunk) <= max_chars:
                            current = candidate_chunk
                        else:
                            if current:
                                chunks.append(current)
                            current = sentence_piece
                    continue

                candidate = f"{current} {sentence}".strip() if current else sentence
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _translate_chunk_with_groq(chunk: str) -> str:
        if not groq_client:
            return chunk

        system_prompt = (
            "You are a professional translator for financial and business journalism. "
            f"Translate the user's English text into {target_language}. "
            "Do not summarize, shorten, or omit details. "
            "Preserve facts, names, numbers, currencies, dates, titles, and ticker symbols. "
            "If a proper noun or ticker should remain in English, keep only that token unchanged while translating the rest. "
            "Keep the tone readable and natural for native readers of the target language. "
            "Return only the translated text with no markdown, no explanations, and no surrounding quotes."
        )

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk},
            ],
        )

        translated = (completion.choices[0].message.content or "").strip()
        if not translated:
            raise ValueError("Empty translation completion")
        return translated.strip("\"'")

    async def _translate_chunk_group(group_chunks: list[str]) -> list[str]:
        if not groq_client:
            raise ValueError("Groq client unavailable")

        payload = {
            "translations": [
                {"index": index, "text": chunk}
                for index, chunk in enumerate(group_chunks)
            ]
        }
        system_prompt = (
            "You are a professional translator for financial and business journalism. "
            f"Translate every item's text into {target_language}. "
            "Do not summarize or omit details. "
            "Preserve names, numbers, currencies, dates, titles, and ticker symbols. "
            "Return ONLY raw JSON with this exact structure: "
            "{\"translations\":[{\"index\":0,\"translated\":\"...\"}]}. "
            "Keep the same indexes. No markdown. No commentary."
        )

        def _call_batch_model() -> list[str]:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            parsed = _extract_json_object(completion.choices[0].message.content or "")
            entries = parsed.get("translations", [])
            if not isinstance(entries, list):
                raise ValueError("Invalid chunk translation payload")

            mapped: dict[int, str] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                index = entry.get("index")
                translated = str(entry.get("translated", "")).strip().strip("\"'")
                if isinstance(index, int) and translated:
                    mapped[index] = translated

            if len(mapped) != len(group_chunks):
                raise ValueError("Chunk translation count mismatch")

            return [mapped[index] for index in range(len(group_chunks))]

        async with _TRANSLATION_SEMAPHORE:
            translated_group = await asyncio.to_thread(_call_batch_model)

        for source_chunk, translated_chunk in zip(group_chunks, translated_group):
            if not _looks_translated(source_chunk, translated_chunk, target_language):
                raise ValueError("Grouped chunk translation appears unchanged")

        return translated_group

    async def _translate_chunk(chunk: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                async with _TRANSLATION_SEMAPHORE:
                    translated_chunk = await asyncio.to_thread(_translate_chunk_with_groq, chunk)
                if not _looks_translated(chunk, translated_chunk, target_language):
                    raise ValueError("Translation output appears unchanged")
                return translated_chunk
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.25 * (attempt + 1))
        raise ValueError(f"Failed to translate chunk after retries: {last_error}")

    max_chunk_chars = 1600 if target_language == "Hindi" else 900
    chunks = _chunk_text_for_translation(cleaned_text, max_chars=max_chunk_chars)

    translated_chunks: list[str] = []
    successful_chunk_count = 0

    chunk_groups = [chunks[i:i + 3] for i in range(0, len(chunks), 3)]

    for group in chunk_groups:
        try:
            translated_group = await _translate_chunk_group(group) if len(group) > 1 else [await _translate_chunk(group[0])]
            translated_chunks.extend(translated_group)
            successful_chunk_count += len(translated_group)
            continue
        except Exception:
            pass

        for chunk in group:
            recovery_pieces = _chunk_text_for_translation(chunk, max_chars=400)
            recovered_parts: list[str] = []

            for piece in recovery_pieces:
                try:
                    recovered_parts.append(await _translate_chunk(piece))
                    successful_chunk_count += 1
                except Exception:
                    recovered_parts.append(piece)

            translated_chunks.append("\n\n".join(part.strip() for part in recovered_parts if part.strip()).strip())

    translated_text = "\n\n".join(chunk.strip() for chunk in translated_chunks if chunk.strip()).strip()
    if not translated_text or successful_chunk_count == 0:
        translated_text = cleaned_text

    result = {
        "original": text,
        "translated": translated_text,
        "target_language": target_language,
        "culturally_adapted": translated_text != cleaned_text,
        "note": (
            f"Translated for {target_language}-speaking readers while preserving the original financial context."
            if translated_text != cleaned_text
            else f"Translation fallback returned original text for {target_language}."
        ),
    }
    if translated_text != cleaned_text:
        _TRANSLATION_CACHE[cache_key] = result
    return result


def _split_sentences(text: str, limit: int = 4) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    return cleaned[:limit]


def _build_free_video_scenes(article: dict) -> list[dict]:
    image_url = article.get("image_url", "")
    summary_sentences = _split_sentences(article.get("summary", ""), limit=2)
    body_sentences = _split_sentences(article.get("full_text") or article.get("content", ""), limit=4)
    tags = article.get("tags", [])[:3]
    tag_text = ", ".join(tags) if tags else "business and markets"

    context_line = summary_sentences[0] if summary_sentences else article.get("summary", "Key developments are unfolding in this story.")
    analysis_line = body_sentences[1] if len(body_sentences) > 1 else body_sentences[0] if body_sentences else article.get("content", "")[:180]
    outlook_line = summary_sentences[1] if len(summary_sentences) > 1 else f"Watch how this story develops across {tag_text}."

    return [
        {
            "timestamp": "0:00",
            "type": "intro",
            "narration": f"Top story from Economic Times: {article['title']}",
            "visual": "Lead headline card with article image and source branding.",
            "duration": 10,
            "image_url": image_url,
        },
        {
            "timestamp": "0:10",
            "type": "context",
            "narration": context_line,
            "visual": f"Context frame highlighting {tag_text} with the article visual.",
            "duration": 18,
            "image_url": image_url,
        },
        {
            "timestamp": "0:28",
            "type": "analysis",
            "narration": analysis_line or "The report highlights the key drivers behind the move.",
            "visual": "Analysis overlay with callouts, key phrases, and supporting article imagery.",
            "duration": 22,
            "image_url": image_url,
        },
        {
            "timestamp": "0:50",
            "type": "data",
            "narration": f"Key themes in focus include {tag_text}.",
            "visual": "Data board with thematic chips, supporting highlights, and summary metrics.",
            "duration": 18,
            "image_url": image_url,
        },
        {
            "timestamp": "1:08",
            "type": "outlook",
            "narration": outlook_line,
            "visual": "Closing outlook card with next-watch points and related story prompts.",
            "duration": 14,
            "image_url": image_url,
        },
        {
            "timestamp": "1:22",
            "type": "outro",
            "narration": "That was your ET Nexus visual briefing powered by Economic Times reporting.",
            "visual": "Branded outro with article thumbnail, source attribution, and return-to-feed cue.",
            "duration": 8,
            "image_url": image_url,
        },
    ]


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "ET Nexus API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "GET  /api/feed?persona={general|investor|founder|student}",
            "POST /api/admin/ingest",
            "POST /api/briefing",
            "POST /api/chat",
            "POST /api/translate",
        ],
    }


@app.post("/api/admin/ingest")
async def admin_ingest():
    """Trigger live ET Markets ingestion and replace the in-memory articles DB."""
    result = await ingest_live_news()
    return {
        "status": "ok",
        **result,
    }


@app.on_event("startup")
async def startup_ingest() -> None:
    """Best-effort initial ingest so the first feed render is not empty."""
    try:
        await _ensure_articles_loaded()
    except Exception:
        # Keep startup resilient; feed endpoint retries lazy ingest.
        pass


@app.get("/api/feed")
async def get_feed(
    persona: str = Query("general", description="User persona: general, investor, founder, or student"),
    refresh: bool = Query(False, description="Force a refresh of the live feed."),
):
    """Return articles filtered by persona relevance."""
    await _ensure_articles_loaded(force_refresh=refresh)

    persona_normalized = persona.lower()
    valid_personas = ["general", "investor", "founder", "student"]
    if persona_normalized not in valid_personas:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid persona '{persona}'. Choose from: {valid_personas}",
        )

    filtered = _filter_articles_for_persona(persona_normalized)

    filtered.sort(
        key=_article_datetime,
        reverse=True,
    )

    # Return a lightweight feed (no full content)
    feed = []
    for a in filtered:
        feed.append({
            "id": a["id"],
            "title": a["title"],
            "summary": a["summary"],
            "author": a["author"],
            "date": a["date"],
            "category": a["category"],
            "tags": a["tags"],
            "sentiment": a["sentiment"],
            "image_url": a["image_url"],
        })

    return {
        "persona": persona_normalized,
        "count": len(feed),
        "articles": feed,
        "last_ingested_at": LAST_INGESTED_AT.isoformat() if LAST_INGESTED_AT else None,
    }


@app.get("/api/article/{article_id}")
async def get_article(article_id: str):
    """Return full article by ID."""
    await _ensure_articles_loaded()
    article = ARTICLES_DB.get(article_id)
    if not article:
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found.")
    return article


@app.post("/api/briefing")
async def get_briefing(request: BriefingRequest):
    """Generate an AI briefing (bullet summary + sentiment) for an article."""
    article = ARTICLES_DB.get(request.article_id)
    if not article:
        raise HTTPException(
            status_code=404,
            detail=f"Article '{request.article_id}' not found.",
        )

    briefing = await generate_briefing(article)
    confidence_score = int(briefing.get("confidence_score", 70))
    confidence_score = max(0, min(100, confidence_score))
    return {
        "article_id": request.article_id,
        "article_title": article["title"],
        "bullets": briefing.get("bullets", []),
        "sentiment": briefing.get("sentiment", "neutral"),
        "confidence_score": confidence_score,
        # Keep legacy numeric contract expected by frontend api.ts.
        "confidence": round(confidence_score / 100, 2),
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Respond to a user question in the context of an article."""
    article = ARTICLES_DB.get(request.context_id)
    if not article:
        raise HTTPException(
            status_code=404,
            detail=f"Article '{request.context_id}' not found.",
        )

    result = await generate_chat_response(request.question, request.context_id)
    return {
        "question": request.question,
        "context_id": request.context_id,
        **result,
    }


@app.post("/api/translate")
async def translate(request: TranslateRequest):
    """Translate and culturally adapt text to a target language."""
    result = await generate_translation(request.text, request.target_language)
    return result


@app.post("/api/translate/batch")
async def translate_batch(request: BatchTranslateRequest):
    """Translate multiple text blocks to a target language."""
    target_language = request.target_language.strip()

    def _normalize_for_compare(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip().lower()

    def _contains_target_script(value: str, language: str) -> bool:
        script_range = _LANGUAGE_SCRIPT_RANGES.get(language)
        if not script_range:
            return True
        start, end = script_range
        return any(start <= char <= end for char in value)

    def _looks_translated(source_text: str, translated_text: str, language: str) -> bool:
        source_norm = _normalize_for_compare(source_text)
        translated_norm = _normalize_for_compare(translated_text)
        if not translated_norm:
            return False
        if translated_norm == source_norm and len(source_norm) > 24:
            return False
        if language in _LANGUAGE_SCRIPT_RANGES and len(source_norm) > 24:
            return _contains_target_script(translated_text, language)
        return True

    async def _translate_group_with_groq(group_items: list[tuple[int, str]]) -> dict[int, str]:
        if not groq_client:
            raise ValueError("Groq client unavailable")

        payload = {
            "translations": [
                {"index": index, "text": text}
                for index, text in group_items
            ]
        }
        system_prompt = (
            "You are a professional translator for financial and business journalism. "
            f"Translate every item's text into {target_language}. "
            "Do not summarize or omit details. "
            "Preserve names, numbers, currencies, dates, titles, and ticker symbols. "
            "Return ONLY raw JSON with this exact structure: "
            "{\"translations\":[{\"index\":0,\"translated\":\"...\"}]}. "
            "Keep the same indexes. No markdown. No commentary."
        )

        def _call_model() -> dict[int, str]:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            parsed = _extract_json_object(completion.choices[0].message.content or "")
            entries = parsed.get("translations", [])
            if not isinstance(entries, list):
                raise ValueError("Invalid batch translation payload")

            mapped: dict[int, str] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                index = entry.get("index")
                translated = str(entry.get("translated", "")).strip().strip("\"'")
                if isinstance(index, int) and translated:
                    mapped[index] = translated
            return mapped

        async with _TRANSLATION_SEMAPHORE:
            mapped = await asyncio.to_thread(_call_model)

        if len(mapped) != len(group_items):
            raise ValueError("Batch translation count mismatch")

        for index, source_text in group_items:
            translated_text = mapped.get(index, "")
            if not _looks_translated(source_text, translated_text, target_language):
                raise ValueError("Batch translation output appears unchanged")

        return mapped

    translations: list[Optional[dict]] = [None] * len(request.texts)
    pending: list[tuple[int, str]] = []

    for index, text in enumerate(request.texts):
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            translations[index] = {
                "original": text,
                "translated": "",
                "target_language": target_language,
                "culturally_adapted": False,
                "note": "Empty text received.",
            }
            continue

        if target_language == "English":
            translations[index] = {
                "original": text,
                "translated": cleaned_text,
                "target_language": target_language,
                "culturally_adapted": False,
                "note": "English selected; original text returned.",
            }
            continue

        cache_key = (target_language, hashlib.sha1(cleaned_text.encode("utf-8")).hexdigest())
        cached = _TRANSLATION_CACHE.get(cache_key)
        if cached:
            translations[index] = cached
            continue

        pending.append((index, text))

    if pending:
        groups: list[list[tuple[int, str]]] = []
        current_group: list[tuple[int, str]] = []
        current_chars = 0

        for item in pending:
            item_chars = len(item[1])
            if current_group and (len(current_group) >= 8 or current_chars + item_chars > 5000):
                groups.append(current_group)
                current_group = []
                current_chars = 0
            current_group.append(item)
            current_chars += item_chars

        if current_group:
            groups.append(current_group)

        for group in groups:
            mapped: dict[int, str] = {}
            try:
                mapped = await _translate_group_with_groq(group)
            except Exception:
                mapped = {}

            for index, source_text in group:
                translated_text = mapped.get(index)
                if translated_text:
                    result = {
                        "original": source_text,
                        "translated": translated_text,
                        "target_language": target_language,
                        "culturally_adapted": True,
                        "note": f"Translated for {target_language}-speaking readers while preserving the original financial context.",
                    }
                    translations[index] = result
                    cache_key = (target_language, hashlib.sha1(source_text.strip().encode("utf-8")).hexdigest())
                    _TRANSLATION_CACHE[cache_key] = result
                else:
                    translations[index] = await generate_translation(source_text, target_language)

    return {
        "target_language": target_language,
        "count": len(request.texts),
        "translations": translations,
    }


# ─────────────────────────────────────────────
# AI News Video Studio
# ─────────────────────────────────────────────

@app.post("/api/video")
async def generate_video(request: VideoRequest):
    """
    Build a free storyboard-style video response from article data.
    """
    article = ARTICLES_DB.get(request.article_id)
    if not article:
        raise HTTPException(status_code=404, detail=f"Article '{request.article_id}' not found.")

    await asyncio.sleep(0.6)
    scenes = _build_free_video_scenes(article)
    key_players = _extract_key_players_from_text(" ".join([
        article.get("title", ""),
        article.get("summary", ""),
        article.get("content", ""),
    ]))

    return {
        "article_id": request.article_id,
        "title": article["title"],
        "video_status": "generated",
        "duration_seconds": sum(scene["duration"] for scene in scenes),
        "format": "mp4",
        "resolution": "1080p",
        "cover_image_url": article.get("image_url", ""),
        "source_url": article.get("link", ""),
        "script": " ".join(scene["narration"] for scene in scenes),
        "scenes": scenes,
        "ai_narration_voice": "professional-female",
        "background_music": "corporate-ambient",
        "data_visuals": [
            {"type": "bar_chart", "label": "Theme Weight", "data": [len(article.get("tags", [])) * 8 or 16, len(key_players) * 10 or 20, max(20, min(90, len(article.get("summary", "")) // 4))]},
            {"type": "line_graph", "label": "Narrative Momentum", "data": [18, 26, 41, 55, 63, 72]},
            {"type": "pie_chart", "label": "Coverage Breakdown", "data": [{"label": tag.title(), "value": max(10, 40 - idx * 8)} for idx, tag in enumerate(article.get("tags", [])[:3])] or [{"label": "Markets", "value": 45}, {"label": "Business", "value": 35}, {"label": "Policy", "value": 20}]},
        ],
        "tags": article["tags"],
        "sentiment": article["sentiment"],
    }


# ─────────────────────────────────────────────
# Story Arc Tracker
# ─────────────────────────────────────────────

@app.get("/api/story-arcs")
async def list_story_arcs(
    persona: str = Query("general", description="User persona: general, investor, founder, or student"),
):
    """List story arcs generated from the current live article set."""
    await _ensure_articles_loaded()

    seed_articles = _filter_articles_for_persona(persona)
    seed_articles.sort(key=_article_datetime, reverse=True)
    arcs = []
    for article in seed_articles[:MAX_STORY_ARC_SUMMARIES]:
        related_articles = _get_related_articles(article, persona=persona)
        arcs.append(_build_story_arc_summary(article, related_articles))
    return {"arcs": arcs}


@app.get("/api/story-arc/{arc_id}")
async def get_story_arc(
    arc_id: str,
    persona: str = Query("general", description="User persona: general, investor, founder, or student"),
):
    """Get an article-dependent story arc built from retrieved related articles."""
    await _ensure_articles_loaded()

    seed_article = ARTICLES_DB.get(arc_id)
    if not seed_article:
        raise HTTPException(status_code=404, detail=f"Story arc '{arc_id}' not found.")

    related_articles = _get_related_articles(seed_article, persona=persona)
    return await _build_story_arc_with_groq(seed_article, related_articles)


# ─────────────────────────────────────────────
# News Navigator — Multi-Article Synthesis
# ─────────────────────────────────────────────

@app.post("/api/navigator")
async def news_navigator(request: NavigatorRequest):
    """
    Synthesize multiple articles on a topic into an explorable briefing.
    TODO: Replace with real LLM multi-document summarisation.
    """
    await asyncio.sleep(1.5)  # Simulated processing

    topic_lower = request.topic.lower()

    # Find related articles by keyword matching
    related = []
    for article in ARTICLES_DB.values():
        title_lower = article["title"].lower()
        tags_lower = [t.lower() for t in article["tags"]]
        if any(kw in title_lower or kw in " ".join(tags_lower) for kw in topic_lower.split()):
            # Filter by persona if specified
            if request.persona in article["persona_relevance"]:
                related.append(article)

    if not related:
        # Fallback: return all articles for the persona
        related = [a for a in ARTICLES_DB.values() if request.persona in a.get("persona_relevance", [])]

    return {
        "topic": request.topic,
        "persona": request.persona,
        "article_count": len(related),
        "synthesis": {
            "headline": f"Deep Briefing: {request.topic}",
            "executive_summary": f"Across {len(related)} articles, ET Nexus identifies converging trends in {request.topic}. Key themes include technological disruption, regulatory response, and market recalibration. The overall sentiment leans {'bullish' if sum(1 for a in related if a['sentiment']=='positive') > len(related)//2 else 'mixed'}.",
            "key_findings": [
                f"📊 {len(related)} related articles analysed spanning {related[0]['date'] if related else 'N/A'} to {related[-1]['date'] if related else 'N/A'}.",
                "🔍 Primary narrative: technological innovation is outpacing regulatory frameworks.",
                "📈 Sentiment trend: shifting from cautious to cautiously optimistic.",
                "⚡ Contrarian view: rapid AI adoption may introduce new systemic risks.",
                "🎯 Investor action: position for regulatory clarity in Q3-Q4 2026.",
            ],
            "follow_up_questions": [
                "How do these developments affect emerging market investors?",
                "What are the second-order effects on traditional financial institutions?",
                "Which regulatory frameworks are most likely to be adopted globally?",
                "How should portfolio allocation change in response to these trends?",
            ],
        },
        "source_articles": [
            {"id": a["id"], "title": a["title"], "sentiment": a["sentiment"], "date": a["date"]}
            for a in related
        ],
    }
