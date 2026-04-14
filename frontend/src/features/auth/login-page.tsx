// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useRef } from "react";
import { useTheme } from "next-themes";
import { cn } from "@/utils/cn";
import { LoginForm } from "./login-form";

const TAGS = [
  { label: "Deployment", icon: "⬡", delay: "0s" },
  { label: "Multi-Channel", icon: "⬡", delay: "0.8s" },
  { label: "Evolution", icon: "◬", delay: "0.6s" },
  { label: "SFT / RL", icon: "◇", delay: "1.2s" },
  { label: "Datasets", icon: "☁", delay: "2.1s" },
  { label: "Tools & Skills", icon: "⚡", delay: "1.8s" },
  { label: "Experts", icon: "⚡", delay: "1.4s" },
  { label: "Analytics", icon: "⊡", delay: "2.4s" },
  { label: "Evaluations", icon: "◈", delay: "3.0s" },
  { label: "Cloud-native", icon: "☁", delay: "3.6s" },
  { label: "Observability", icon: "☁", delay: "2.1s" },
  { label: "Data Hub", icon: "⊕", delay: "4.2s" },
] as const;

export function LoginPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { resolvedTheme, setTheme } = useTheme();

  /* ── animated grid background ── */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let animId: number;
    let time = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth * 2;
      canvas.height = canvas.offsetHeight * 2;
      ctx.scale(2, 2);
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      const w = canvas.offsetWidth;
      const h = canvas.offsetHeight;
      ctx.clearRect(0, 0, w, h);

      const dark = document.documentElement.classList.contains("dark");
      const am = dark ? 1 : 2.5;

      const gridSize = 40;
      const cols = Math.ceil(w / gridSize) + 1;
      const rows = Math.ceil(h / gridSize) + 1;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = c * gridSize;
          const y = r * gridSize;
          const dist = Math.sqrt((x - w * 0.4) ** 2 + (y - h * 0.45) ** 2);
          const wave = Math.sin(dist * 0.008 - time * 0.6) * 0.5 + 0.5;
          ctx.fillStyle = `rgba(245,158,11,${(0.03 + wave * 0.08) * am})`;
          ctx.beginPath();
          ctx.arc(x, y, 1, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      for (let r = 0; r < rows; r++) {
        const y = r * gridSize;
        const wave = Math.sin(y * 0.02 - time * 0.3) * 0.5 + 0.5;
        ctx.strokeStyle = `rgba(245,158,11,${(0.015 + wave * 0.02) * am})`;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      for (let c = 0; c < cols; c++) {
        const x = c * gridSize;
        const wave = Math.sin(x * 0.015 - time * 0.2) * 0.5 + 0.5;
        ctx.strokeStyle = `rgba(245,158,11,${(0.01 + wave * 0.015) * am})`;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }

      const orbX = w * 0.4 + Math.sin(time * 0.3) * 60;
      const orbY = h * 0.45 + Math.cos(time * 0.2) * 40;
      const g1 = ctx.createRadialGradient(orbX, orbY, 0, orbX, orbY, 200);
      g1.addColorStop(0, `rgba(245,158,11,${0.06 * am})`);
      g1.addColorStop(0.5, `rgba(245,158,11,${0.02 * am})`);
      g1.addColorStop(1, "rgba(245,158,11,0)");
      ctx.fillStyle = g1;
      ctx.fillRect(0, 0, w, h);

      const orb2X = w * 0.7 + Math.cos(time * 0.25) * 80;
      const orb2Y = h * 0.6 + Math.sin(time * 0.35) * 50;
      const g2 = ctx.createRadialGradient(orb2X, orb2Y, 0, orb2X, orb2Y, 180);
      g2.addColorStop(0, `rgba(59,130,246,${0.04 * am})`);
      g2.addColorStop(1, "rgba(59,130,246,0)");
      ctx.fillStyle = g2;
      ctx.fillRect(0, 0, w, h);

      time += 0.016;
      animId = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className="font-mono bg-background text-foreground h-screen flex overflow-hidden text-sm leading-normal antialiased">
      {/* ═══ LEFT PANEL — Brand + Animation ═══ */}
      <div className="flex-1 relative overflow-hidden flex flex-col justify-between p-12 px-14">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* vignette */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse at 40% 45%, transparent 30%, var(--background) 80%)",
          }}
        />

        {/* hat decoration — lower-right corner */}
        <div
          className={cn(
            "absolute bottom-24 -right-8 z-1 opacity-0",
            "animate-fade-in",
          )}
          style={{ animationDelay: "0.5s" }}
        >
          <div
            className="w-[420px] h-[420px] opacity-100 dark:opacity-35 pointer-events-none select-none"
            style={{
              backgroundColor: "var(--hat-fill)",
              maskImage: "url(/hat.svg)",
              WebkitMaskImage: "url(/hat.svg)",
              maskSize: "contain",
              WebkitMaskSize: "contain",
              maskRepeat: "no-repeat",
              WebkitMaskRepeat: "no-repeat",
              maskPosition: "center",
              WebkitMaskPosition: "center",
              filter: "drop-shadow(0 0 80px rgba(245, 158, 11, 0.2))",
            }}
          />
        </div>

        {/* logo + brand */}
        <div
          className={cn(
            "relative z-10 opacity-0",
            "animate-slide-right",
          )}
        >
          <div className="flex items-center gap-4 mb-14">
            <img
              src="/login.svg"
              alt="Surogate"
              className="w-14 h-14 rounded-2xl animate-logo-glow"
            />
            <div>
              <div className="font-display font-extrabold text-2xl text-foreground tracking-tight">
                Surogate
              </div>
              <div className="font-mono text-sm text-muted-foreground tracking-[0.12em] uppercase">
                Build Reliable Agents
              </div>
            </div>
          </div>

          <h2 className="mt-20 font-display text-7xl font-extrabold text-foreground leading-[1.1] tracking-tight max-w-[650px]">
            Build, train, 
            <br />
            and deploy 
            <br />
            <span className="text-primary">reliable Agents</span>
          </h2>
          <p className="font-display text-2xl text-muted-foreground mt-6 max-w-[50rem] leading-relaxed">
            Create agents that solve complex tasks, adapt to new situations, and continuously evolve.
          </p>
        </div>

        {/* capability tags */}
        <div
          className={cn("relative z-10 opacity-0", "animate-fade-in")}
          style={{ animationDelay: "0.3s" }}
        >
          <div className="flex flex-wrap gap-2 max-w-[60rem]">
            {TAGS.map((tag) => (
              <div
                key={tag.label}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-muted/50 border border-border backdrop-blur animate-float-tag"
                style={{ animationDelay: tag.delay }}
              >
                <span className="text-sm text-primary">{tag.icon}</span>
                <span className="font-display text-sm text-subtle">
                  {tag.label}
                </span>
              </div>
            ))}
          </div>
          <div className="font-display mt-6 text-sm text-faint">
            <a className="underline" href="https://surogate.ai" target="_blank">surogate.ai</a> &middot; Copyright © 2026 <a className="underline" href="https://invergent.ai" target="_blank">Invergent SA</a>
          </div>
        </div>
      </div>

      {/* ═══ RIGHT PANEL — Login Form ═══ */}
      <div className="w-[480px] min-w-[480px] flex flex-col justify-center items-center px-14 py-12 border-l border-line bg-card relative">
        {/* theme toggle */}
        <button
          type="button"
          onClick={() =>
            setTheme(resolvedTheme === "dark" ? "light" : "dark")
          }
          className="absolute top-6 right-6 w-9 h-9 rounded-lg border border-border bg-muted flex items-center justify-center cursor-pointer text-muted-foreground transition-colors duration-150 hover:text-foreground hover:border-primary/30"
          aria-label="Toggle theme"
        >
          {resolvedTheme === "dark" ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="5" />
              <line x1="12" y1="1" x2="12" y2="3" />
              <line x1="12" y1="21" x2="12" y2="23" />
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
              <line x1="1" y1="12" x2="3" y2="12" />
              <line x1="21" y1="12" x2="23" y2="12" />
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>

        {/* scan line */}
        <div
          className="absolute left-0 right-0 h-px pointer-events-none animate-scan"
          style={{
            background:
              "linear-gradient(90deg, transparent, rgba(245,158,11,0.08), transparent)",
          }}
        />

        <div
          className={cn(
            "w-full max-w-[360px] opacity-0",
            "animate-fade-up",
          )}
          style={{ animationDelay: "0.15s" }}
        >
          <LoginForm />
        </div>
      </div>
    </div>
  );
}
