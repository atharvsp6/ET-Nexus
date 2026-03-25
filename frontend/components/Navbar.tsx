"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAppContext } from "@/lib/context";
import {
  Zap,
  ChevronDown,
  Globe,
  Briefcase,
  GraduationCap,
  TrendingUp,
  Film,
  Compass,
  Navigation,
} from "lucide-react";

const personas = [
  { key: "general" as const, label: "General", icon: Globe },
  { key: "investor" as const, label: "Investor", icon: TrendingUp },
  { key: "founder" as const, label: "Founder", icon: Briefcase },
  { key: "student" as const, label: "Student", icon: GraduationCap },
];

const languages = ["English", "Hindi", "Tamil", "Telugu", "Bengali"] as const;

const navLinks = [
  { href: "/", label: "Feed", icon: Zap },
  { href: "/navigator", label: "Navigator", icon: Navigation },
  { href: "/video", label: "Video Studio", icon: Film },
  { href: "/story-arc", label: "Story Arc", icon: Compass },
];

export default function Navbar() {
  const { persona, setPersona, language, setLanguage } = useAppContext();
  const [personaOpen, setPersonaOpen] = useState(false);
  const [langOpen, setLangOpen] = useState(false);
  const personaRef = useRef<HTMLDivElement>(null);
  const langRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (personaRef.current && !personaRef.current.contains(e.target as Node)) {
        setPersonaOpen(false);
      }
      if (langRef.current && !langRef.current.contains(e.target as Node)) {
        setLangOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const currentPersona = personas.find((p) => p.key === persona)!;
  const PersonaIcon = currentPersona.icon;

  return (
    <nav className="glass sticky top-0 z-50 border-b border-[var(--color-border)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo + Nav Links */}
          <div className="flex items-center gap-6">
            <Link href="/" className="flex items-center gap-2 group">
              <div className="relative">
                <Zap
                  className="text-[var(--color-accent)] transition-all duration-300 group-hover:drop-shadow-[0_0_8px_var(--color-accent-glow)]"
                  size={28}
                  strokeWidth={2.5}
                />
              </div>
              <span className="text-xl font-bold tracking-tight">
                <span className="gradient-text">ET</span>{" "}
                <span className="text-[var(--color-foreground)]">Nexus</span>
              </span>
              <span className="hidden sm:inline-block text-[10px] font-mono font-medium text-[var(--color-accent)] bg-[var(--color-accent-glow)] px-2 py-0.5 rounded-full border border-[var(--color-accent)]/30 ml-1">
                AI
              </span>
            </Link>

            {/* Navigation Links */}
            <div className="hidden md:flex items-center gap-1">
              {navLinks.map((link) => {
                const Icon = link.icon;
                const active = pathname === link.href;
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                      active
                        ? "bg-[var(--color-accent)]/10 text-[var(--color-accent)] border border-[var(--color-accent)]/20"
                        : "text-[var(--color-muted)] hover:text-[var(--color-foreground)] hover:bg-[var(--color-surface)]"
                    }`}
                  >
                    <Icon size={13} />
                    {link.label}
                  </Link>
                );
              })}
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3">
            {/* Persona Switcher */}
            <div className="relative" ref={personaRef}>
              <button
                onClick={() => { setPersonaOpen(!personaOpen); setLangOpen(false); }}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] hover:border-[var(--color-accent)]/50 transition-all duration-200 text-sm font-medium cursor-pointer"
              >
                <PersonaIcon size={16} className="text-[var(--color-accent)]" />
                <span className="hidden sm:inline">{currentPersona.label}</span>
                <ChevronDown
                  size={14}
                  className={`text-[var(--color-muted)] transition-transform duration-200 ${personaOpen ? "rotate-180" : ""}`}
                />
              </button>
              {personaOpen && (
                <div className="absolute right-0 top-full mt-2 w-48 rounded-xl bg-[var(--color-surface)] border border-[var(--color-border)] shadow-2xl shadow-black/40 overflow-hidden animate-fade-in">
                  <div className="px-3 py-2 border-b border-[var(--color-border)]">
                    <p className="text-[10px] uppercase tracking-widest text-[var(--color-muted)] font-semibold">
                      Select Persona
                    </p>
                  </div>
                  {personas.map((p) => {
                    const Icon = p.icon;
                    const active = p.key === persona;
                    return (
                      <button
                        key={p.key}
                        onClick={() => { setPersona(p.key); setPersonaOpen(false); }}
                        className={`w-full flex items-center gap-3 px-3 py-2.5 text-sm transition-colors cursor-pointer ${
                          active
                            ? "bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                            : "hover:bg-[var(--color-surface-hover)] text-[var(--color-foreground)]"
                        }`}
                      >
                        <Icon size={16} />
                        <span className="font-medium">{p.label}</span>
                        {active && (
                          <span className="ml-auto w-2 h-2 rounded-full bg-[var(--color-accent)]" />
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Language Toggle */}
            <div className="relative" ref={langRef}>
              <button
                onClick={() => { setLangOpen(!langOpen); setPersonaOpen(false); }}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--color-surface)] border border-[var(--color-border)] hover:border-[var(--color-accent)]/50 transition-all duration-200 text-sm font-medium cursor-pointer"
              >
                <Globe size={16} className="text-[var(--color-accent-secondary)]" />
                <span className="hidden sm:inline">{language}</span>
                <ChevronDown
                  size={14}
                  className={`text-[var(--color-muted)] transition-transform duration-200 ${langOpen ? "rotate-180" : ""}`}
                />
              </button>
              {langOpen && (
                <div className="absolute right-0 top-full mt-2 w-40 rounded-xl bg-[var(--color-surface)] border border-[var(--color-border)] shadow-2xl shadow-black/40 overflow-hidden animate-fade-in">
                  {languages.map((l) => (
                    <button
                      key={l}
                      onClick={() => { setLanguage(l); setLangOpen(false); }}
                      className={`w-full flex items-center gap-2 px-3 py-2.5 text-sm transition-colors cursor-pointer ${
                        l === language
                          ? "bg-[var(--color-accent-secondary)]/10 text-[var(--color-accent-secondary)]"
                          : "hover:bg-[var(--color-surface-hover)] text-[var(--color-foreground)]"
                      }`}
                    >
                      <span className="font-medium">{l}</span>
                      {l === language && (
                        <span className="ml-auto w-2 h-2 rounded-full bg-[var(--color-accent-secondary)]" />
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Mobile Nav */}
        <div className="flex md:hidden items-center gap-1 pb-2 overflow-x-auto">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all duration-200 ${
                  active
                    ? "bg-[var(--color-accent)]/10 text-[var(--color-accent)]"
                    : "text-[var(--color-muted)] hover:text-[var(--color-foreground)]"
                }`}
              >
                <Icon size={12} />
                {link.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
