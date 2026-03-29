# Deployment Guide (ET Nexus)

This project is a two-service app:
- Backend: FastAPI (`backend/`)
- Frontend: Next.js (`frontend/`)

Recommended production setup:
- Deploy backend to Render (or Railway)
- Deploy frontend to Vercel

## 1) Backend Deployment (Render)

### Create service
- New Web Service from this repository
- Root Directory: `backend`
- Runtime: Python 3.10+

### Build and start commands
- Build Command:

```bash
pip install -r requirements.txt
```

- Start Command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Required environment variables
Set these in Render:

- `GROQ_API_KEY` = your Groq key
- `GROQ_MODEL` = `llama-3.1-8b-instant` (or preferred model)
- `CORS_ALLOW_ORIGINS` = your deployed frontend origin (comma-separated if multiple)

Example:

```env
CORS_ALLOW_ORIGINS=https://your-frontend.vercel.app
```

### Verify backend
Open:
- `https://your-backend-domain.onrender.com/`

Expected response:

```json
{"name":"ET Nexus API","version":"0.1.0","status":"running"}
```

## 2) Frontend Deployment (Vercel)

### Import project
- New Project from this repository
- Root Directory: `frontend`
- Framework: Next.js (auto-detected)

### Build settings
Defaults are usually correct:
- Build command: `npm run build`
- Output: Next.js default

### Required environment variable
Set in Vercel Project Settings -> Environment Variables:

- `API_BASE_URL` = your backend URL

Recommended optional public override:

- `NEXT_PUBLIC_API_BASE_URL` = your backend URL (only needed if you want browser-side direct calls)

Example:

```env
API_BASE_URL=https://your-backend-domain.onrender.com
```

Deploy and open your Vercel URL.

## 3) Post-deploy checks

1. Open frontend homepage and confirm feed loads.
2. Open an article page and confirm briefing/chat requests succeed.
3. If requests fail due to CORS, re-check backend `CORS_ALLOW_ORIGINS`.
4. If AI responses fail, re-check backend `GROQ_API_KEY` and model.

## 4) Local production-like run (optional)

Backend:

```bash
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run build
npm run start
```

Set frontend env before start:

```env
API_BASE_URL=http://localhost:8000
```

## 5) Optional single-platform deploy with Docker

If you prefer containerized deploys, use two services with these Dockerfiles.

Backend Dockerfile (`backend/Dockerfile`):

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

Frontend Dockerfile (`frontend/Dockerfile`):

```Dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
EXPOSE 3000
CMD ["npm", "run", "start"]
```

Use platform-level environment variables for API URL, CORS, and Groq credentials.
