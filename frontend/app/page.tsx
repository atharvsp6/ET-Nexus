"use client";

import React, { useEffect, useState } from "react";
import { useAppContext } from "@/lib/context";
import { fetchFeed, Article, translateTexts } from "@/lib/api";
import NewsCard from "@/components/NewsCard";
import StoryArc from "@/components/StoryArc";
import {
  Newspaper,
  Loader2,
  AlertCircle,
  TrendingUp,
  Briefcase,
  GraduationCap,
  Sparkles,
  BarChart3,
  Activity,
} from "lucide-react";

const personaMeta = {
  general: {
    icon: Newspaper,
    greeting: "General Briefing",
    subtitle: "A cross-persona stream of markets, startups, technology, and education updates.",
  },
  investor: {
    icon: TrendingUp,
    greeting: "Market Intelligence",
    subtitle: "Curated insights for portfolio managers, analysts, and active investors.",
  },
  founder: {
    icon: Briefcase,
    greeting: "Founder's Briefing",
    subtitle: "Strategy, funding, and competitive intelligence for builders.",
  },
  student: {
    icon: GraduationCap,
    greeting: "Learning Lab",
    subtitle: "Deep dives into concepts, trends, and frameworks shaping modern finance.",
  },
};

export default function Dashboard() {
  const { persona, language } = useAppContext();
  const [articles, setArticles] = useState<Article[]>([]);
  const [translatedArticles, setTranslatedArticles] = useState<{ language: string; articles: Article[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [translatingFeed, setTranslatingFeed] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadFeed() {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchFeed(persona);
        if (cancelled) return;
        setArticles(data.articles);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unknown feed error");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void loadFeed();
    return () => {
      cancelled = true;
    };
  }, [persona]);

  useEffect(() => {
    let cancelled = false;

    async function loadTranslatedFeed() {
      if (language === "English" || articles.length === 0) {
        setTranslatedArticles(null);
        setTranslatingFeed(false);
        return;
      }

      setTranslatedArticles(null);
      setTranslatingFeed(true);
      try {
        const payload = articles.flatMap((article) => [article.title, article.summary]);
        const res = await translateTexts(payload, language);
        if (cancelled) return;

        const nextArticles = articles.map((article, index) => ({
          ...article,
          title: res.translations[index * 2]?.translated || article.title,
          summary: res.translations[index * 2 + 1]?.translated || article.summary,
        }));
        setTranslatedArticles({ language, articles: nextArticles });
      } catch {
        if (!cancelled) setTranslatedArticles(null);
      } finally {
        if (!cancelled) setTranslatingFeed(false);
      }
    }

    void loadTranslatedFeed();
    return () => {
      cancelled = true;
    };
  }, [articles, language]);

  const meta = personaMeta[persona];
  const PersonaIcon = meta.icon;
  const visibleArticles = translatedArticles?.language === language ? translatedArticles.articles : articles;

  return (
    <div className="space-y-10">
      {/* Hero Header */}
      <header className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[var(--color-surface)] to-[var(--color-background)] border border-[var(--color-border)] p-8 animate-fade-in-up">
        {/* Decorative grid */}
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: `radial-gradient(circle, var(--color-accent) 1px, transparent 1px)`,
          backgroundSize: "24px 24px",
        }} />

        <div className="relative flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/20 flex items-center justify-center">
            <PersonaIcon size={28} className="text-[var(--color-accent)]" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-2xl font-bold">{meta.greeting}</h1>
              <Sparkles size={18} className="text-[var(--color-accent)]" />
            </div>
            <p className="text-sm text-[var(--color-muted)]">
              {meta.subtitle}
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs text-[var(--color-muted)]">
            <div className="flex items-center gap-1.5">
              <BarChart3 size={14} className="text-[var(--color-accent)]" />
              <span className="font-mono">{visibleArticles.length} stories</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Activity size={14} className="text-[var(--color-positive)]" />
              <span className="font-mono">{translatingFeed ? "Translating" : "Live"}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Loading State */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <Loader2 size={32} className="animate-spin text-[var(--color-accent)]" />
          <p className="text-sm text-[var(--color-muted)]">
            Loading your personalised feed...
          </p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-[var(--color-negative)]/10 border border-[var(--color-negative)]/30 text-[var(--color-negative)]">
          <AlertCircle size={20} />
          <div>
            <p className="text-sm font-semibold">Failed to load feed</p>
            <p className="text-xs opacity-80">{error}. Check that your backend is reachable and API base URL settings are correct.</p>
          </div>
        </div>
      )}

      {/* Articles Grid */}
      {!loading && !error && (
        <>
          <section>
            <div className="flex items-center gap-2 mb-6">
              <Newspaper size={18} className="text-[var(--color-accent)]" />
              <h2 className="text-lg font-bold">Today&apos;s Feed</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {visibleArticles.map((article, i) => (
                <NewsCard key={article.id} article={article} index={i} />
              ))}
            </div>
          </section>

          {/* Story Arc */}
          <StoryArc articles={visibleArticles} persona={persona} />
        </>
      )}
    </div>
  );
}
