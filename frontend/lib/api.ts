/**
 * ET Nexus — API Client
 * Communicates with the FastAPI backend.
 */

import type { Persona } from "@/lib/context";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

// ─── Types ───

export interface Article {
  id: string;
  title: string;
  summary: string;
  content?: string;
  author: string;
  date: string;
  category: string;
  tags: string[];
  sentiment: "positive" | "negative" | "neutral";
  image_url: string;
  persona_relevance?: string[];
}

export interface FeedResponse {
  persona: string;
  count: number;
  articles: Article[];
  last_ingested_at?: string | null;
}

export interface BriefingResponse {
  article_id: string;
  article_title: string;
  bullets: string[];
  sentiment: string;
  confidence: number;
}

export interface ChatResponse {
  question: string;
  context_id: string;
  response: string;
  sources: string[];
}

export interface TranslateResponse {
  original: string;
  translated: string;
  target_language: string;
  culturally_adapted: boolean;
  note: string;
}

export interface BatchTranslateResponse {
  target_language: string;
  count: number;
  translations: TranslateResponse[];
}

export interface VideoScene {
  timestamp: string;
  type: string;
  narration: string;
  visual: string;
  duration: number;
  image_url?: string;
}

export interface VideoResponse {
  article_id: string;
  title: string;
  video_status: string;
  duration_seconds: number;
  format: string;
  resolution: string;
  cover_image_url?: string;
  source_url?: string;
  script?: string;
  scenes: VideoScene[];
  ai_narration_voice: string;
  background_music: string;
  data_visuals: { type: string; label: string; data: unknown[] }[];
  tags: string[];
  sentiment: string;
}

export interface StoryArcEvent {
  date: string;
  title: string;
  description: string;
  sentiment: string;
  impact_score: number;
  related_articles: string[];
}

export interface KeyPlayer {
  name: string;
  role: string;
  sentiment: string;
}

export interface StoryArc {
  id: string;
  title: string;
  description: string;
  status: string;
  events: StoryArcEvent[];
  key_players: KeyPlayer[];
  sentiment_trajectory: { date: string; score: number }[];
  predictions: string[];
}

export interface StoryArcSummary {
  id: string;
  title: string;
  description: string;
  status: string;
  event_count: number;
  player_count: number;
}

export interface NavigatorResponse {
  topic: string;
  persona: string;
  article_count: number;
  synthesis: {
    headline: string;
    executive_summary: string;
    key_findings: string[];
    follow_up_questions: string[];
  };
  source_articles: { id: string; title: string; sentiment: string; date: string }[];
}

// ─── API Functions ───

export async function fetchFeed(persona: Persona, refresh = false): Promise<FeedResponse> {
  const params = new URLSearchParams({ persona });
  if (refresh) params.set("refresh", "true");
  const res = await fetch(`${API_BASE}/api/feed?${params.toString()}`);
  if (!res.ok) throw new Error(`Feed fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchArticle(id: string): Promise<Article> {
  const res = await fetch(`${API_BASE}/api/article/${id}`);
  if (!res.ok) throw new Error(`Article fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchBriefing(articleId: string): Promise<BriefingResponse> {
  const res = await fetch(`${API_BASE}/api/briefing`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ article_id: articleId }),
  });
  if (!res.ok) throw new Error(`Briefing fetch failed: ${res.status}`);
  return res.json();
}

export async function sendChat(question: string, contextId: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, context_id: contextId }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json();
}

export async function translateText(text: string, targetLanguage: string): Promise<TranslateResponse> {
  const res = await fetch(`${API_BASE}/api/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, target_language: targetLanguage }),
  });
  if (!res.ok) throw new Error(`Translation failed: ${res.status}`);
  return res.json();
}

export async function translateTexts(texts: string[], targetLanguage: string): Promise<BatchTranslateResponse> {
  const res = await fetch(`${API_BASE}/api/translate/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts, target_language: targetLanguage }),
  });
  if (!res.ok) throw new Error(`Batch translation failed: ${res.status}`);
  return res.json();
}

export async function generateVideo(articleId: string): Promise<VideoResponse> {
  const res = await fetch(`${API_BASE}/api/video`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ article_id: articleId }),
  });
  if (!res.ok) throw new Error(`Video generation failed: ${res.status}`);
  return res.json();
}

export async function fetchStoryArcs(persona: Persona): Promise<{ arcs: StoryArcSummary[] }> {
  const res = await fetch(`${API_BASE}/api/story-arcs?persona=${persona}`);
  if (!res.ok) throw new Error(`Story arcs fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchStoryArc(arcId: string, persona: Persona): Promise<StoryArc> {
  const res = await fetch(`${API_BASE}/api/story-arc/${arcId}?persona=${persona}`);
  if (!res.ok) throw new Error(`Story arc fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchNavigator(topic: string, persona: string): Promise<NavigatorResponse> {
  const res = await fetch(`${API_BASE}/api/navigator`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, persona }),
  });
  if (!res.ok) throw new Error(`Navigator failed: ${res.status}`);
  return res.json();
}
