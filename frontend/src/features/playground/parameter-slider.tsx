// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { cn } from "@/utils/cn";

interface ParameterSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  className?: string;
}

export function ParameterSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  suffix = "",
  className,
}: ParameterSliderProps) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn("mb-3.5", className)}>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="font-display text-[10px] text-muted-foreground">
          {label}
        </span>
        <span className="font-mono text-[11px] font-semibold text-foreground">
          {value}
          {suffix}
        </span>
      </div>
      <div className="relative flex h-[18px] items-center">
        <div className="absolute h-[3px] w-full rounded-sm bg-border">
          <div
            className="h-full rounded-sm bg-primary transition-[width] duration-100 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number.parseFloat(e.target.value))}
          className="absolute m-0 h-[18px] w-full cursor-pointer opacity-0"
        />
        <div
          className="pointer-events-none absolute h-3.5 w-3.5 rounded-full border-2 border-primary bg-card transition-[left] duration-100 ease-out"
          style={{ left: `calc(${pct}% - 7px)` }}
        />
      </div>
    </div>
  );
}
