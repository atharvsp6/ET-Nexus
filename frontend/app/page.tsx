"use client";

import React, { useEffect, useState } from "react";
import { useAppContext } from "@/lib/context";
import { fetchFeed, Article } from "@/lib/api";
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
  const { persona } = useAppContext();
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchFeed(persona)
      .then((data) => {
        setArticles(data.articles);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [persona]);

  const meta = personaMeta[persona];
  const PersonaIcon = meta.icon;

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
              <span className="font-mono">{articles.length} stories</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Activity size={14} className="text-[var(--color-positive)]" />
              <span className="font-mono">Live</span>
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
            <p className="text-xs opacity-80">{error}. Make sure the backend is running on port 8000.</p>
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
              {articles.map((article, i) => (
                <NewsCard key={article.id} article={article} index={i} />
              ))}
            </div>
          </section>

          {/* Story Arc */}
          <StoryArc />
        </>
      )}
    </div>
  );
}
