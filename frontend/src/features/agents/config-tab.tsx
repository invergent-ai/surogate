import { Button } from "@/components/ui/button";
import type { Agent } from "./agents-data";

export function ConfigTab({ agent }: { agent: Agent }) {
  return (
    <div className="animate-in fade-in duration-150 space-y-4">
      {/* Environment variables */}
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Environment Variables
          </span>
          <Button variant="outline" size="xs">
            Edit
          </Button>
        </div>
        <div className="font-mono">
          {agent.env.map((e) => (
            <div
              key={e.key}
              className="px-4 py-2 border-b border-border/50 last:border-b-0 flex items-center gap-4 text-xs"
            >
              <span className="text-blue-500 min-w-[220px]">{e.key}</span>
              <span className="text-muted-foreground/20">=</span>
              <span className="text-foreground/70">{e.value}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Container info */}
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Container Image
          </span>
        </div>
        <div className="p-4">
          <div className="bg-background border border-border rounded-md px-3.5 py-2.5 font-mono text-[11px] text-muted-foreground">
            <span className="text-muted-foreground/40">image:</span>{" "}
            <span className="text-foreground/70">{agent.image}</span>
          </div>
          <div className="grid grid-cols-4 gap-3 mt-3">
            {[
              { label: "CPU REQUEST", value: agent.resources.cpuReq },
              { label: "CPU LIMIT", value: agent.resources.cpuLim },
              { label: "MEM REQUEST", value: agent.resources.memReq },
              { label: "MEM LIMIT", value: agent.resources.memLim },
            ].map((r) => (
              <div key={r.label}>
                <div className="text-[9px] text-muted-foreground/40 mb-0.5 font-display">
                  {r.label}
                </div>
                <code className="text-[11px] text-foreground/70">{r.value}</code>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}