// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useNavigate, useSearch } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { GpuOfferPicker } from "@/components/gpu-offer-picker";
import { ArrowLeft } from "lucide-react";
import { SUPPORTED_PROVIDERS } from "./compute-data";

export function BackendOffersPage() {
  const { backend } = useSearch({ strict: false }) as { backend?: string };
  const navigate = useNavigate();
  const info = SUPPORTED_PROVIDERS.find(p => p.key === backend);

  const goBack = () => navigate({ to: "/studio/compute/cloud" });

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title={info ? `${info.name} — Available Instances` : "Available Instances"}
        subtitle="GPU and CPU instances available from this backend"
      />

      <div className="flex-1 overflow-y-auto px-7 py-5 pb-10">
        <Button
          variant="ghost"
          size="sm"
          className="mb-4 text-faint"
          onClick={goBack}
        >
          <ArrowLeft size={14} />
          Back to Cloud
        </Button>

        {backend ? (
          <GpuOfferPicker backend={backend} selectedOffer={null} onSelect={() => {}} />
        ) : (
          <div className="text-sm text-faint">No backend specified.</div>
        )}
      </div>
    </div>
  );
}
