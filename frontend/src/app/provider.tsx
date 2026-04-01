// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "next-themes";
import type { ReactNode } from "react";

interface AppProviderProps {
  children: ReactNode;
}

export function AppProvider({ children }: AppProviderProps) {
  return (
    <ThemeProvider attribute="class" defaultTheme="light">
      {children}
      <Toaster position="bottom-right" visibleToasts={2} expand={true} />
    </ThemeProvider>
  );
}