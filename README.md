# ET Nexus

ET Nexus is an AI-assisted business news product built around live `Economic Times` coverage.

It combines a `FastAPI` backend for ingestion and AI services with a `Next.js` frontend for a persona-aware newsroom experience featuring translated content, dynamic story arcs, article chat, topic navigation, and a free storyboard-style video view.

## Features

- Live ET-only news feed from multiple Economic Times RSS sources
- Persona-aware views for `general`, `investor`, `founder`, and `student`
- Stable article IDs and automatic feed refresh based on TTL
- Real ET article thumbnails extracted from article pages
- Full article page with AI briefing and article-grounded chat
- Dynamic article-linked story arcs instead of static canned arcs
- Topic-based navigator for multi-article synthesis
- Multi-language translation for feed items, article content, and briefing bullets
- Free article-backed video storyboard experience using ET imagery

## Tech Stack

### Frontend

- `Next.js 16`
- `React 19`
- `TypeScript`
- `Tailwind CSS 4`

### Backend

- `FastAPI`
- `Uvicorn`
- `Pydantic`
- `Groq`
- `feedparser`
- `BeautifulSoup4`
- `requests`

## Project Structure

```text
ET-Nexus/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── package.json
└── README.md
```

## Main Product Areas

### Feed

- Persona-filtered ET article stream
- Real article images
- Dynamic home-page story arc preview

### Article Page

- Full article content
- AI briefing
- Article-grounded chat
- Translation-aware rendering
- Inline video preview

### Story Arc Tracker

- Article-dependent story arc timeline
- Key players
- Sentiment trajectory
- Predictions / what-to-watch-next

### Navigator

- Topic search across the current ET article corpus
- Synthesis, findings, source articles, and follow-up questions

### Video Studio

- Article selector
- Storyboard-style scenes
- Article-backed scene visuals
- Data-visual metadata

## Setup

## Prerequisites

- `Python 3.10+`
- `Node.js 18+`
- `npm`
- a valid `GROQ_API_KEY`

## Backend

```bash
cd backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Optional tuning
FEED_REFRESH_TTL_SECONDS=900
MAX_FEED_ARTICLES=12
MAX_STORY_ARC_SUMMARIES=6
MAX_RELATED_ARTICLES=4
INGEST_WITH_GROQ=false
TRANSLATION_CONCURRENCY=2
```

Run the backend:

```bash
cd backend
. .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

Backend URL:

- `http://127.0.0.1:8000`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

- `http://localhost:3000`

## Useful Commands

### Frontend

```bash
cd frontend
npm run dev
npm run lint
npm run build
```

### Backend

```bash
cd backend
. .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Syntax Check

```bash
python3 -m py_compile backend/main.py
```

## Environment Variables

Required:

- `GROQ_API_KEY`

Optional:

- `GROQ_MODEL`
- `CORS_ALLOW_ORIGINS`
- `FEED_REFRESH_TTL_SECONDS`
- `MAX_FEED_ARTICLES`
- `MAX_STORY_ARC_SUMMARIES`
- `MAX_RELATED_ARTICLES`
- `INGEST_WITH_GROQ`
- `TRANSLATION_CONCURRENCY`
- `NEXT_PUBLIC_API_BASE_URL` (frontend)

See deployment steps in `DEPLOYMENT.md`.

## Backend API

### Core

- `GET /`
- `POST /api/admin/ingest`
- `GET /api/feed?persona={persona}&refresh={bool}`
- `GET /api/article/{article_id}`

### AI Features

- `POST /api/briefing`
- `POST /api/chat`
- `POST /api/translate`
- `POST /api/translate/batch`
- `POST /api/video`
- `GET /api/story-arcs?persona={persona}`
- `GET /api/story-arc/{arc_id}?persona={persona}`
- `POST /api/navigator`

## Current Notes

- The app uses ET content only.
- Articles and translation cache are stored in memory, so restarting the backend clears runtime state.
- Translation quality depends on a working Groq model connection.
- The current video system is a free storyboard/video-preview pipeline, not a rendered avatar video generator.
- Some AI surfaces are stronger than others: story arcs are more dynamic than navigator synthesis today.

## Known Limitations

- No persistent database yet
- No automated end-to-end test suite yet
- Some secondary UI labels are still English
- Very large full-article translations can be slower in some languages
- Image rendering still uses plain `<img>` tags in a few places, which causes non-blocking Next.js lint warnings

## Recommended Next Improvements

1. Add persistent storage for articles and translations.
2. Improve navigator synthesis with stronger multi-document reasoning.
3. Localize more of the UI chrome and generated surfaces.
4. Replace plain image tags with `next/image`.
5. Add focused backend and frontend tests.
6. Upgrade the free storyboard video system to a true rendered media pipeline if paid infra becomes acceptable.

## Quick Start

1. Set up the backend virtual environment.
2. Install backend dependencies.
3. Add `GROQ_API_KEY` to `backend/.env`.
4. Start the backend on port `8000`.
5. Install frontend dependencies.
6. Start the frontend on port `3000`.
7. Open `http://localhost:3000`.

## Summary

ET Nexus is a working local prototype for an AI-native ET newsroom with:

- live ET ingestion
- article-backed images
- persona-aware feed filtering
- dynamic story arcs
- topic navigation
- translation support
- free video/storyboard previews
