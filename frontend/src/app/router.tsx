// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { createRouter } from "@tanstack/react-router";
import { Route as rootRoute } from "./routes/__root";
import { Route as loginRoute } from "./routes/login";
import { Route as onboardingRoute } from "./routes/onboarding";
import { Route as changePasswordRoute } from "./routes/change-password";
import {
  Route as studioRoute,
  studioIndexRoute,
  agentsRoute,
  modelsRoute,
  skillsRoute,
  skillsIndexRoute,
  skillsToolsRoute,
  skillsMcpRoute,
  computeRoute,
  computeIndexRoute,
  computeClusterNodesRoute,
  computeCloudRoute,
  computeWorkloadQueueRoute,
  computeCostsRoute,
  computePoliciesRoute,
  connectCloudRoute,
  backendOffersRoute,
  monitoringRoute,
  conversationsRoute,
  evaluationsRoute,
  playgroundRoute,
  datasetsRoute,
  trainingRoute,
  hubRoute,
  hubIndexRoute,
  repoDetailRoute,
  settingsRoute,
  settingsIndexRoute,
  settingsProjectsRoute,
  settingsApiKeysRoute,
  settingsHubRoute,
  settingsIntegrationsRoute,
  settingsNotificationsRoute,
} from "./routes/studio";
import { Route as indexRoute } from "./routes/index";

const routeTree = rootRoute.addChildren([
  indexRoute,
  onboardingRoute,
  loginRoute,
  changePasswordRoute,
  studioRoute.addChildren([
    studioIndexRoute,
    agentsRoute,
    modelsRoute,
    skillsRoute.addChildren([
      skillsIndexRoute,
      skillsToolsRoute,
      skillsMcpRoute,
    ]),
    computeRoute.addChildren([
      computeIndexRoute,
      computeClusterNodesRoute,
      computeCloudRoute,
      computeWorkloadQueueRoute,
      computeCostsRoute,
      computePoliciesRoute,
    ]),
    connectCloudRoute,
    backendOffersRoute,
    monitoringRoute,
    conversationsRoute,
    evaluationsRoute,
    playgroundRoute,
    datasetsRoute,
    trainingRoute,
    hubRoute.addChildren([hubIndexRoute, repoDetailRoute]),
    settingsRoute.addChildren([
      settingsIndexRoute,
      settingsProjectsRoute,
      settingsApiKeysRoute,
      settingsHubRoute,
      settingsIntegrationsRoute,
      settingsNotificationsRoute,
    ]),
  ]),
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
