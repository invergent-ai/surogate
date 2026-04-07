// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table, TableHeader, TableBody, TableHead, TableRow, TableCell,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAppStore } from "@/stores/app-store";
import { cn } from "@/utils/cn";
import type { InstanceOffer } from "@/api/compute";

function formatMemory(mib: number): string {
  return mib >= 1024 ? `${(mib / 1024).toFixed(0)} GB` : `${mib} MB`;
}

export function GpuOfferPicker({
  backend,
  selectedOffer,
  onSelect,
}: {
  backend: string;
  selectedOffer: InstanceOffer | null;
  onSelect: (offer: InstanceOffer) => void;
}) {
  const backendOffers = useAppStore((s) => s.backendOffers);
  const backendOffersLoading = useAppStore((s) => s.backendOffersLoading);
  const backendOffersError = useAppStore((s) => s.backendOffersError);
  const fetchOffers = useAppStore((s) => s.fetchBackendOffers);
  const [spotOnly, setSpotOnly] = useState(false);

  const offers = backendOffers.filter(
    (o) => o.backend === backend && (!spotOnly || o.spot),
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          onClick={fetchOffers}
          disabled={backendOffersLoading}
        >
          {backendOffersLoading ? "Loading offers\u2026" : "Fetch available Instances"}
        </Button>
        <label className="flex items-center gap-1.5 cursor-pointer ml-2">
          <Checkbox
            checked={spotOnly}
            onCheckedChange={(v) => setSpotOnly(v === true)}
          />
          <span className="text-sm text-foreground font-display">
            Spot instances
          </span>
          <span className="text-[9px] text-muted-foreground">
            (cheaper but may be terminated anytime)
          </span>
        </label>
        {selectedOffer && (
          <span className="text-[10px] text-muted-foreground ml-auto">
            Selected: {selectedOffer.gpu_count}&times; {selectedOffer.gpu_name} &mdash; {selectedOffer.instance} ({selectedOffer.region})
          </span>
        )}
      </div>

      {backendOffersError && (
        <div className="text-sm text-destructive">
          {backendOffersError}
        </div>
      )}

      {offers.length > 0 && (
        <ScrollArea className="h-[40vh] rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead >Instance</TableHead>
                <TableHead >GPU</TableHead>
                <TableHead >vCPUs</TableHead>
                <TableHead >RAM</TableHead>
                <TableHead >Region</TableHead>
                <TableHead className="text-[10px] text-right">$/hr</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {offers.map((o) => {
                const key = `${o.instance}-${o.region}-${o.spot}`;
                const isSelected = selectedOffer
                  && selectedOffer.instance === o.instance
                  && selectedOffer.region === o.region
                  && selectedOffer.spot === o.spot;
                return (
                  <TableRow
                    key={key}
                    className={cn(
                      "cursor-pointer text-xs border-primary-foreground/20",
                      isSelected && "bg-primary/10",
                    )}
                    onClick={() => onSelect(o)}
                  >
                    <TableCell className="font-medium ">{o.instance}</TableCell>
                    <TableCell className="">
                      {o.gpu_count > 0 ? `${o.gpu_count}\u00d7 ${o.gpu_name}` : "\u2014"}
                      {o.gpu_memory_mib ? ` (${formatMemory(o.gpu_memory_mib)})` : ""}
                    </TableCell>
                    <TableCell className="">{o.cpus}</TableCell>
                    <TableCell className="">{formatMemory(o.memory_mib)}</TableCell>
                    <TableCell className="">{o.region}</TableCell>
                    <TableCell className=" text-right font-mono">
                      ${o.price.toFixed(2)}
                      {o.spot && <span className="ml-1 text-green-500 text-[9px]">spot</span>}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </ScrollArea>
      )}
    </div>
  );
}
