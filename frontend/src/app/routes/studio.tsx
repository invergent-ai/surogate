// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Outlet, createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { useInitialData } from "@/hooks/use-initial-data";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const StudioPage = lazy(() =>
  import("@/features/studio/studio-page").then((m) => ({
    default: m.StudioPage,
  })),
);

const AgentsPage = lazy(() =>
  import("@/features/agents/agents-page").then((m) => ({
    default: m.AgentsPage,
  })),
);

const ModelsPage = lazy(() =>
  import("@/features/models/models-page").then((m) => ({
    default: m.ModelsPage,
  })),
);

const SkillsPage = lazy(() =>
  import("@/features/skills/skills-page").then((m) => ({
    default: m.SkillsPage,
  })),
);

const SkillsTab = lazy(() =>
  import("@/features/skills/skills-tab").then((m) => ({
    default: m.SkillsTab,
  })),
);

const ToolsTab = lazy(() =>
  import("@/features/skills/tools-tab").then((m) => ({
    default: m.ToolsTab,
  })),
);

const McpServersTab = lazy(() =>
  import("@/features/skills/mcp-servers-tab").then((m) => ({
    default: m.McpServersTab,
  })),
);

const ComputePage = lazy(() =>
  import("@/features/compute/compute-page").then((m) => ({
    default: m.ComputePage,
  })),
);

const OverviewTab = lazy(() =>
  import("@/features/compute/overview-tab").then((m) => ({
    default: m.OverviewTab,
  })),
);

const ClusterNodesTab = lazy(() =>
  import("@/features/compute/cluster-nodes-tab").then((m) => ({
    default: m.ClusterNodesTab,
  })),
);

const CloudTab = lazy(() =>
  import("@/features/compute/cloud-tab").then((m) => ({
    default: m.CloudTab,
  })),
);

const WorkloadQueueTab = lazy(() =>
  import("@/features/compute/workload-queue-tab").then((m) => ({
    default: m.WorkloadQueueTab,
  })),
);

const CostsTab = lazy(() =>
  import("@/features/compute/costs-tab").then((m) => ({
    default: m.CostsTab,
  })),
);

const PoliciesTab = lazy(() =>
  import("@/features/compute/policies-tab").then((m) => ({
    default: m.PoliciesTab,
  })),
);

const MonitoringPage = lazy(() =>
  import("@/features/monitoring/monitoring-page").then((m) => ({
    default: m.MonitoringPage,
  })),
);

const ConversationsPage = lazy(() =>
  import("@/features/conversations/conversations-page").then((m) => ({
    default: m.ConversationsPage,
  })),
);

const EvaluationsPage = lazy(() =>
  import("@/features/evaluations/evaluations-page").then((m) => ({
    default: m.EvaluationsPage,
  })),
);

const PlaygroundPage = lazy(() =>
  import("@/features/playground/playground-page").then((m) => ({
    default: m.PlaygroundPage,
  })),
);

const DatasetsPage = lazy(() =>
  import("@/features/datasets/datasets-page").then((m) => ({
    default: m.DatasetsPage,
  })),
);

const TrainingPage = lazy(() =>
  import("@/features/training/training-page").then((m) => ({
    default: m.TrainingPage,
  })),
);

const HubPage = lazy(() =>
  import("@/features/hub/hub-page").then((m) => ({
    default: m.HubPage,
  })),
);

const RepoDetailPage = lazy(() =>
  import("@/features/hub/repo-detail-page").then((m) => ({
    default: m.RepoDetailPage,
  })),
);

const SettingsPage = lazy(() =>
  import("@/features/settings/settings-page").then((m) => ({
    default: m.SettingsPage,
  })),
);

const ProfileTab = lazy(() =>
  import("@/features/settings/profile-tab").then((m) => ({
    default: m.ProfileTab,
  })),
);

const SettingsProjectsTab = lazy(() =>
  import("@/features/settings/projects-tab").then((m) => ({
    default: m.ProjectsTab,
  })),
);

const ApiKeysTab = lazy(() =>
  import("@/features/settings/api-keys-tab").then((m) => ({
    default: m.ApiKeysTab,
  })),
);

const HubConfigTab = lazy(() =>
  import("@/features/settings/hub-tab").then((m) => ({
    default: m.HubTab,
  })),
);

const SettingsIntegrationsTab = lazy(() =>
  import("@/features/settings/integrations-tab").then((m) => ({
    default: m.IntegrationsTab,
  })),
);

const NotificationsTab = lazy(() =>
  import("@/features/settings/notifications-tab").then((m) => ({
    default: m.NotificationsTab,
  })),
);

function StudioLayout() {
  useInitialData();
  return <Outlet />;
}

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/studio",
  beforeLoad: () => requireAuth(),
  component: StudioLayout,
});

export const studioIndexRoute = createRoute({
  getParentRoute: () => Route,
  path: "/",
  component: StudioPage,
});

export const agentsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/agents",
  component: AgentsPage,
});

export const modelsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/models",
  component: ModelsPage,
});

export const skillsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/skills",
  component: SkillsPage,
});

export const skillsIndexRoute = createRoute({
  getParentRoute: () => skillsRoute,
  path: "/",
  component: SkillsTab,
});

export const skillsToolsRoute = createRoute({
  getParentRoute: () => skillsRoute,
  path: "/tools",
  component: ToolsTab,
});

export const skillsMcpRoute = createRoute({
  getParentRoute: () => skillsRoute,
  path: "/mcp",
  component: McpServersTab,
});

export const computeRoute = createRoute({
  getParentRoute: () => Route,
  path: "/compute",
  component: ComputePage,
});

export const computeIndexRoute = createRoute({
  getParentRoute: () => computeRoute,
  path: "/",
  component: OverviewTab,
});

export const computeClusterNodesRoute = createRoute({
  getParentRoute: () => computeRoute,
  path: "/cluster-nodes",
  component: ClusterNodesTab,
});

export const computeCloudRoute = createRoute({
  getParentRoute: () => computeRoute,
  path: "/cloud",
  component: CloudTab,
});

export const computeWorkloadQueueRoute = createRoute({
  getParentRoute: () => computeRoute,
  path: "/workload-queue",
  component: WorkloadQueueTab,
});

export const computeCostsRoute = createRoute({
  getParentRoute: () => computeRoute,
  path: "/costs",
  component: CostsTab,
});

export const computePoliciesRoute = createRoute({
  getParentRoute: () => computeRoute,
  path: "/policies",
  component: PoliciesTab,
});

export const monitoringRoute = createRoute({
  getParentRoute: () => Route,
  path: "/monitoring",
  component: MonitoringPage,
});

export const conversationsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/conversations",
  component: ConversationsPage,
});

export const evaluationsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/evaluations",
  component: EvaluationsPage,
});

export const playgroundRoute = createRoute({
  getParentRoute: () => Route,
  path: "/playground",
  component: PlaygroundPage,
});

export const datasetsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/datasets",
  component: DatasetsPage,
});

export const trainingRoute = createRoute({
  getParentRoute: () => Route,
  path: "/training",
  component: TrainingPage,
});

export const hubRoute = createRoute({
  getParentRoute: () => Route,
  path: "/hub",
  component: Outlet,
});

export const hubIndexRoute = createRoute({
  getParentRoute: () => hubRoute,
  path: "/",
  component: HubPage,
});

function RepoDetailWrapper() {
  const { _splat } = repoDetailRoute.useParams();
  return <RepoDetailPage repoId={_splat} />;
}

export const repoDetailRoute = createRoute({
  getParentRoute: () => hubRoute,
  path: "$",
  component: RepoDetailWrapper,
});

export const settingsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/settings",
  component: SettingsPage,
});

export const settingsIndexRoute = createRoute({
  getParentRoute: () => settingsRoute,
  path: "/",
  component: ProfileTab,
});

export const settingsProjectsRoute = createRoute({
  getParentRoute: () => settingsRoute,
  path: "/projects",
  component: SettingsProjectsTab,
});

export const settingsApiKeysRoute = createRoute({
  getParentRoute: () => settingsRoute,
  path: "/api-keys",
  component: ApiKeysTab,
});

export const settingsHubRoute = createRoute({
  getParentRoute: () => settingsRoute,
  path: "/hub",
  component: HubConfigTab,
});

export const settingsIntegrationsRoute = createRoute({
  getParentRoute: () => settingsRoute,
  path: "/integrations",
  component: SettingsIntegrationsTab,
});

export const settingsNotificationsRoute = createRoute({
  getParentRoute: () => settingsRoute,
  path: "/notifications",
  component: NotificationsTab,
});
