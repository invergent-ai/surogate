import { cn } from "@/utils/cn";
import type { Agent } from "./agents-data";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export function VersionsTab({ agent }: { agent: Agent }) {
  return (
    <div className="animate-in fade-in duration-150">
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center gap-2">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Version History
          </span>
          <span className="text-[9px] px-1.5 py-px bg-muted rounded text-muted-foreground">
            {agent.versions.length} versions
          </span>
        </div>
        {agent.versions.map((v, i) => (
          <div
            key={v.version}
            className="px-4 py-3 border-b border-border/50 last:border-b-0 flex items-start gap-3 hover:bg-muted/30 transition-colors cursor-pointer"
          >
            {/* Timeline */}
            <div className="flex flex-col items-center pt-1 w-4 shrink-0">
              <div
                className={cn(
                  "w-2.5 h-2.5 rounded-full shrink-0 border-2",
                  v.status === "active"
                    ? "bg-green-500 border-green-500"
                    : v.status === "deploying"
                      ? "bg-amber-500 border-amber-500"
                      : v.status === "error"
                        ? "bg-red-500 border-red-500"
                        : "bg-muted border-muted-foreground/30",
                )}
              />
              {i < agent.versions.length - 1 && (
                <div className="w-px flex-1 bg-border min-h-[20px] mt-1" />
              )}
            </div>

            <div className="flex-1">
              <div className="flex items-center gap-2 mb-0.5">
                <code
                  className={cn(
                    "text-xs",
                    v.status === "active"
                      ? "text-foreground font-semibold"
                      : "text-muted-foreground",
                  )}
                >
                  v{v.version}
                </code>
                {v.status === "active" && (
                  <Badge variant="active">LIVE</Badge>
                )}
                {v.status === "deploying" && (
                  <Badge variant="active">DEPLOYING</Badge>
                )}
                {v.status === "error" && (
                  <Badge variant="danger">FAILED</Badge>
                )}
                <code className="text-[9px] text-muted-foreground/40">
                  {v.hash}
                </code>
              </div>
              <div className="text-[11px] text-foreground/70 mb-0.5">
                {v.change}
              </div>
              <div className="text-[9px] text-muted-foreground/40">
                {v.author} &middot; {v.date}
              </div>
            </div>

            <div className="shrink-0 flex gap-1">
              {v.status === "previous" && (
                <Button variant="outline" size="xs">
                  Rollback
                </Button>
              )}
              <Button variant="ghost" size="xs">
                Diff
              </Button>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}