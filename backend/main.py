"""
ET Nexus — The AI-Native Newsroom
FastAPI Backend with mock data and simulated AI endpoints.
"""

import asyncio
import json
import os
import random
import re
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from typing import Optional

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
ET_RSS_URLS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/2146842.cms",
    "https://economictimes.indiatimes.com/tech/rssfeeds/13352306.cms",
    "https://b2b.economictimes.indiatimes.com/rss/startup",
    "https://education.economictimes.indiatimes.com/rss/topstories",
]
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
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


async def _ensure_articles_loaded() -> None:
    if ARTICLES_DB:
        return

    async with INGEST_LOCK:
        if ARTICLES_DB:
            return
        await ingest_live_news()


async def ingest_live_news() -> dict:
    entries = []
    for rss_url in ET_RSS_URLS:
        feed = await asyncio.to_thread(feedparser.parse, rss_url)
        if hasattr(feed, "entries"):
            entries.extend(feed.entries)

    random.shuffle(entries)

    valid_entries = []
    for entry in entries:
        title = _clean_text(entry.get("title", ""))
        description = _clean_text(BeautifulSoup(entry.get("description", ""), "html.parser").get_text(" ", strip=True))
        if not title:
            continue
        if "live updates" in title.lower():
            continue
        if not description:
            continue
        valid_entries.append(entry)
        if len(valid_entries) >= 12:
            break

    sentiment_map = {
        "bullish": "positive",
        "bearish": "negative",
        "neutral": "neutral",
    }

    ingested = {}
    for entry in valid_entries:
        link = entry.get("link", "")
        if not link:
            continue

        try:
            response = await asyncio.to_thread(requests.get, link, headers=_REQUEST_HEADERS, timeout=15)
            response.raise_for_status()
            full_text = _extract_article_text(response.text)
        except Exception:
            continue

        if not full_text:
            continue

        title = _clean_text(entry.get("title", "Economic Times Market Update"))
        llm_meta = await _classify_article_with_groq(full_text, title)

        article_id = str(uuid.uuid4())
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
            "image_url": "",
            "persona_relevance": [persona],
            "source": "Economic Times",
            "read_time": read_time,
            "link": link,
        }
        ingested[article_id] = article

    ARTICLES_DB.clear()
    ARTICLES_DB.update(ingested)

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
    """
    Simulate a culturally adapted translation.
    TODO: Replace with real translation API (e.g., Google Translate + LLM post-editing).
    """
    await asyncio.sleep(0.5)  # Simulated latency

    translations = {
        "Hindi": {
            "Markets crashed today": "आज बाज़ारों में भारी गिरावट आई — निवेशकों में बेचैनी का माहौल है।",
            "default": "यह एक AI-जनित हिंदी अनुवाद है। मूल पाठ का सांस्कृतिक रूप से अनुकूलित संस्करण यहाँ प्रदर्शित होगा।",
        },
        "Tamil": {
            "default": "இது AI-உருவாக்கிய தமிழ் மொழிபெயர்ப்பு. கலாச்சார ரீதியாக தழுவிய பதிப்பு இங்கே காட்டப்படும். வணிக நிதி செய்திகளை எளிய தமிழில் புரிந்துகொள்ளலாம்.",
        },
        "Telugu": {
            "default": "ఇది AI-రూపొందించిన తెలుగు అనువాదం. సాంస్కృతికంగా అనుకూలమైన వెర్షన్ ఇక్కడ చూపబడుతుంది. వ్యాపార ఆర్థిక వార్తలను సరళమైన తెలుగులో అర్థం చేసుకోవచ్చు.",
        },
        "Bengali": {
            "default": "এটি একটি AI-উৎপন্ন বাংলা অনুবাদ। সাংস্কৃতিকভাবে অভিযোজিত সংস্করণ এখানে প্রদর্শিত হবে। ব্যবসায়িক অর্থনৈতিক সংবাদ সহজ বাংলায় বোঝা যাবে।",
        },
    }

    lang_map = translations.get(target_language, translations.get("Hindi"))
    translated_text = lang_map.get(text, lang_map.get("default", f"[Translated to {target_language}]: {text}"))

    return {
        "original": text,
        "translated": translated_text,
        "target_language": target_language,
        "culturally_adapted": True,
        "note": f"Translation adapted for {target_language}-speaking audience with local idioms and context.",
    }


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
async def get_feed(persona: str = Query("general", description="User persona: general, investor, founder, or student")):
    """Return articles filtered by persona relevance."""
    await _ensure_articles_loaded()

    persona_normalized = persona.lower()
    valid_personas = ["general", "investor", "founder", "student"]
    if persona_normalized not in valid_personas:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid persona '{persona}'. Choose from: {valid_personas}",
        )

    if persona_normalized == "general":
        filtered = list(ARTICLES_DB.values())
    else:
        filtered = [
            article for article in ARTICLES_DB.values()
            if persona_normalized in article.get("persona_relevance", [])
        ]

    filtered.sort(
        key=lambda article: datetime.strptime(article.get("date", "1900-01-01"), "%Y-%m-%d")
        if article.get("date") else datetime.min,
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

    return {"persona": persona_normalized, "count": len(feed), "articles": feed}


@app.get("/api/article/{article_id}")
async def get_article(article_id: str):
    """Return full article by ID."""
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


# ─────────────────────────────────────────────
# AI News Video Studio
# ─────────────────────────────────────────────

@app.post("/api/video")
async def generate_video(request: VideoRequest):
    """
    Simulate AI video generation from an article.
    TODO: Replace with real video generation pipeline.
    """
    article = ARTICLES_DB.get(request.article_id)
    if not article:
        raise HTTPException(status_code=404, detail=f"Article '{request.article_id}' not found.")

    await asyncio.sleep(2.0)  # Simulated video generation time

    return {
        "article_id": request.article_id,
        "title": article["title"],
        "video_status": "generated",
        "duration_seconds": 90,
        "format": "mp4",
        "resolution": "1080p",
        "scenes": [
            {
                "timestamp": "0:00",
                "type": "intro",
                "narration": f"Breaking: {article['title']}",
                "visual": "Animated title card with ET Nexus branding and category badge",
                "duration": 8,
            },
            {
                "timestamp": "0:08",
                "type": "context",
                "narration": article["summary"],
                "visual": "Animated data visualization showing key metrics and trends",
                "duration": 20,
            },
            {
                "timestamp": "0:28",
                "type": "analysis",
                "narration": f"Key analysis from {article['author']} highlights several critical factors driving this development.",
                "visual": "Split-screen with expert quote overlay and supporting charts",
                "duration": 25,
            },
            {
                "timestamp": "0:53",
                "type": "data",
                "narration": "Let's look at the numbers that matter.",
                "visual": "Animated bar charts and line graphs with real-time data overlays",
                "duration": 20,
            },
            {
                "timestamp": "1:13",
                "type": "outlook",
                "narration": "Looking ahead, analysts expect this trend to reshape the landscape significantly.",
                "visual": "Forward-looking prediction graphic with confidence intervals",
                "duration": 12,
            },
            {
                "timestamp": "1:25",
                "type": "outro",
                "narration": "Stay informed with ET Nexus — your AI-native newsroom.",
                "visual": "ET Nexus logo with subscribe CTA and related article thumbnails",
                "duration": 5,
            },
        ],
        "ai_narration_voice": "professional-female",
        "background_music": "corporate-ambient",
        "data_visuals": [
            {"type": "bar_chart", "label": "Market Impact", "data": [45, 72, 38, 91, 65]},
            {"type": "line_graph", "label": "Trend Over Time", "data": [12, 19, 35, 42, 58, 71, 85]},
            {"type": "pie_chart", "label": "Sector Breakdown", "data": [{"label": "Tech", "value": 35}, {"label": "Finance", "value": 28}, {"label": "Energy", "value": 22}, {"label": "Other", "value": 15}]},
        ],
        "tags": article["tags"],
        "sentiment": article["sentiment"],
    }


# ─────────────────────────────────────────────
# Story Arc Tracker
# ─────────────────────────────────────────────

STORY_ARCS = {
    "market-contagion": {
        "id": "market-contagion",
        "title": "The Ripple Effects of Market Contagion",
        "description": "Tracking how a single bank's real-estate losses cascaded into a global financial episode.",
        "status": "active",
        "events": [
            {
                "date": "2026-03-03",
                "title": "Nasdaq Flash Anomaly",
                "description": "Nasdaq-100 futures plunge 4.1% in 12 minutes before snapping back. Algorithmic cascades identified as primary amplifier.",
                "sentiment": "negative",
                "impact_score": 7.2,
                "related_articles": ["3"],
            },
            {
                "date": "2026-03-10",
                "title": "Vereinigte Kreditbank Disclosure",
                "description": "German lender reveals €6.2B real-estate loss. Global sell-off begins within hours — Asian CDS spreads widen 120 bps.",
                "sentiment": "negative",
                "impact_score": 9.5,
                "related_articles": ["1"],
            },
            {
                "date": "2026-03-15",
                "title": "SEC Kill Switch Proposal",
                "description": "SEC proposes Rule 15c3-7 mandating automated circuit-breakers for all algorithmic trading systems.",
                "sentiment": "neutral",
                "impact_score": 6.8,
                "related_articles": ["3"],
            },
            {
                "date": "2026-03-18",
                "title": "Synthetic Data Hedge",
                "description": "Major hedge funds reveal they anticipated the contagion using synthetic data stress-testing and avoided 80% of drawdown.",
                "sentiment": "positive",
                "impact_score": 5.4,
                "related_articles": ["5"],
            },
            {
                "date": "2026-03-22",
                "title": "Central Banks Deploy AI Simulations",
                "description": "Five central banks pilot LLM-driven agent-based models to stress-test monetary policy before implementation.",
                "sentiment": "positive",
                "impact_score": 7.8,
                "related_articles": ["7"],
            },
        ],
        "key_players": [
            {"name": "Vereinigte Kreditbank AG", "role": "Origin of contagion", "sentiment": "negative"},
            {"name": "SEC", "role": "Regulatory response", "sentiment": "neutral"},
            {"name": "BIS (Bank for International Settlements)", "role": "Research & analysis", "sentiment": "neutral"},
            {"name": "Citadel Securities", "role": "Industry voice — supportive of Kill Switch", "sentiment": "positive"},
            {"name": "Two Sigma / DE Shaw", "role": "Synthetic data pioneers — hedged successfully", "sentiment": "positive"},
        ],
        "sentiment_trajectory": [
            {"date": "2026-03-03", "score": -0.6},
            {"date": "2026-03-10", "score": -0.9},
            {"date": "2026-03-15", "score": -0.3},
            {"date": "2026-03-18", "score": 0.2},
            {"date": "2026-03-22", "score": 0.5},
        ],
        "predictions": [
            "Basel IV capital buffer revision expected by Q3 2026.",
            "SEC Kill Switch Rule likely to pass with modifications by year-end.",
            "LLM-based monetary policy simulations to become standard at G7 central banks within 18 months.",
            "Algo-trading market share may temporarily decline 5-8% as firms retool for compliance.",
        ],
    },
    "ai-finance-revolution": {
        "id": "ai-finance-revolution",
        "title": "AI's Takeover of Financial Markets",
        "description": "From earnings prediction to monetary policy — AI is reshaping every corner of finance.",
        "status": "active",
        "events": [
            {
                "date": "2026-03-08",
                "title": "UPI Hits 20B Transactions",
                "description": "India's digital payments infrastructure matures to support AI-driven fintech layers.",
                "sentiment": "positive",
                "impact_score": 6.5,
                "related_articles": ["4"],
            },
            {
                "date": "2026-03-12",
                "title": "Deep Learning Beats Wall Street",
                "description": "QuantLens AI's transformer achieves 78% accuracy on earnings surprises.",
                "sentiment": "positive",
                "impact_score": 8.7,
                "related_articles": ["2"],
            },
            {
                "date": "2026-03-18",
                "title": "Synthetic Data Goes Mainstream",
                "description": "GANs and diffusion models generate synthetic market data for unprecedented stress testing.",
                "sentiment": "positive",
                "impact_score": 7.3,
                "related_articles": ["5"],
            },
            {
                "date": "2026-03-22",
                "title": "Central Banks Go AI-Native",
                "description": "Five central banks pilot LLM-powered economic simulations for policy decisions.",
                "sentiment": "positive",
                "impact_score": 8.9,
                "related_articles": ["7"],
            },
            {
                "date": "2026-03-24",
                "title": "AI-Native Startup Hits Unicorn",
                "description": "PayLoop reaches $50M ARR and $1.2B valuation with autonomous AI agents.",
                "sentiment": "positive",
                "impact_score": 7.1,
                "related_articles": ["8"],
            },
        ],
        "key_players": [
            {"name": "QuantLens AI", "role": "Pioneering deep-learning earnings prediction", "sentiment": "positive"},
            {"name": "Two Sigma / DE Shaw", "role": "Leading synthetic data adoption", "sentiment": "positive"},
            {"name": "RBI / BOE / ECB / BOJ / Fed", "role": "Central banks pioneering AI simulations", "sentiment": "neutral"},
            {"name": "PayLoop (Aisha Patel)", "role": "AI-native fintech unicorn", "sentiment": "positive"},
            {"name": "NPCI (India)", "role": "Enabling AI-driven payments infrastructure", "sentiment": "positive"},
        ],
        "sentiment_trajectory": [
            {"date": "2026-03-08", "score": 0.6},
            {"date": "2026-03-12", "score": 0.8},
            {"date": "2026-03-18", "score": 0.7},
            {"date": "2026-03-22", "score": 0.8},
            {"date": "2026-03-24", "score": 0.9},
        ],
        "predictions": [
            "AI-driven trading strategies to manage >$2T in assets by end of 2026.",
            "At least 3 more AI-native fintechs expected to reach unicorn status this year.",
            "Regulatory frameworks for AI in finance to emerge across G20 nations.",
            "Synthetic data market for finance to exceed $5B by 2028.",
        ],
    },
}


@app.get("/api/story-arcs")
async def list_story_arcs():
    """List all available story arcs."""
    arcs = []
    for arc in STORY_ARCS.values():
        arcs.append({
            "id": arc["id"],
            "title": arc["title"],
            "description": arc["description"],
            "status": arc["status"],
            "event_count": len(arc["events"]),
            "player_count": len(arc["key_players"]),
        })
    return {"arcs": arcs}


@app.get("/api/story-arc/{arc_id}")
async def get_story_arc(arc_id: str):
    """Get full story arc with timeline, key players, sentiment, and predictions."""
    await asyncio.sleep(0.8)  # Simulated processing
    arc = STORY_ARCS.get(arc_id)
    if not arc:
        raise HTTPException(status_code=404, detail=f"Story arc '{arc_id}' not found.")
    return arc


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
