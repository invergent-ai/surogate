// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useNavigate, useSearch } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { SUPPORTED_PROVIDERS, PROVIDER_COLORS } from "./compute-data";
import { ConnectAws } from "./components/connect-aws";
import { ConnectGcp } from "./components/connect-gcp";
import { ConnectAzure } from "./components/connect-azure";
import { ConnectNebius } from "./components/connect-nebius";
import { ConnectRunpod } from "./components/connect-runpod";
import { useTheme } from "@/hooks/use-theme";

const PROVIDER_FORMS: Record<string, React.ComponentType<{ onCancel: () => void }>> = {
  aws: ConnectAws,
  gcp: ConnectGcp,
  azure: ConnectAzure,
  runpod: ConnectRunpod,
  nebius: ConnectNebius,
};

export function ConnectCloudPage() {
  const { provider } = useSearch({ strict: false }) as { provider?: string };
  const navigate = useNavigate();
  const info = SUPPORTED_PROVIDERS.find(p => p.key === provider);
  const { isDark } = useTheme();
  const logo = isDark ? info?.logoDark : info?.logoWhite;
  const Form = provider ? PROVIDER_FORMS[provider] : undefined;

  const goBack = () => navigate({ to: "/studio/compute/cloud" });

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title={info ? `Connect ${info.name}` : "Connect Cloud Provider"}
        subtitle={info?.description}
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

        <Card size="sm" className="p-6">
          {info && (
            <div className="flex items-center gap-3 border-b border-line">
              {logo
                ? <img src={logo} alt={info.name} className="w-25 h-15 shrink-0 object-contain" />
                : <span className="text-lg" style={{ color: PROVIDER_COLORS[info.key] }}>{"\u2601"}</span>
              }
              <div>
                <div className="text-[11px] text-faint">{info.description}</div>
              </div>
            </div>
          )}

          {Form ? (
            <Form onCancel={goBack} />
          ) : (
            <div className="text-sm text-faint">Unknown provider. Please go back and select a supported cloud.</div>
          )}
        </Card>
      </div>
    </div>
  );
}
