"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";

export type Persona = "general" | "investor" | "founder" | "student";
export type Language = "English" | "Hindi" | "Tamil" | "Telugu" | "Bengali";

interface AppContextType {
  persona: Persona;
  setPersona: (p: Persona) => void;
  language: Language;
  setLanguage: (l: Language) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [persona, setPersona] = useState<Persona>("general");
  const [language, setLanguage] = useState<Language>("English");

  return (
    <AppContext.Provider value={{ persona, setPersona, language, setLanguage }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
}
