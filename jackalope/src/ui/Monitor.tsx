import React from "react";
import { Box, Text } from "ink";
import type { FeedStatus } from "../feed.ts";
import { C } from "./theme.ts";
import { ShimmerText } from "./Shimmer.tsx";

/** The loss panel body: the rendered chart string, or a status placeholder with
 *  guidance — the #1 confusion is a training that isn't writing the feed jackalope
 *  watches (needs `report_to: [surogate]`, and the path must match). */
export function Chart({ image, feedState, feedPath }: { image: string; feedState?: FeedStatus; feedPath?: string }) {
  if (image) return <Text>{image}</Text>;
  if (feedState === "unavailable") {
    return (
      <Box flexDirection="column">
        <Text color={C.warm}>feed unavailable — nothing was written to</Text>
        <Text color={C.text}>{"  "}{feedPath ?? "the feed path"}</Text>
        <Box marginTop={1} flexDirection="column">
          <Text color={C.muted}>1 · add to your training config:</Text>
          <Text color={C.dim}>{"     report_to: [surogate]"}</Text>
          <Text color={C.muted}>2 · point jackalope at the run's feed:</Text>
          <Text color={C.dim}>{"     jackalope <surogate_metrics_path>   (or set $SUROGATE_METRICS_PATH)"}</Text>
        </Box>
      </Box>
    );
  }
  return (
    <Box flexDirection="column">
      <ShimmerText text="waiting for the first loss values…" />
      <Text color={C.dim}>(the run must set `report_to: [surogate]`)</Text>
    </Box>
  );
}
