import React from "react";
import { Box, Text } from "ink";
import { C } from "./theme.ts";

/** A clean, borderless section: a small uppercase title + content, separated by
 *  whitespace (the PostHog-wizard look — no heavy boxes). */
export function Panel({
  title,
  children,
  flexGrow,
  height,
  width,
  minHeight,
}: {
  title: string;
  children: React.ReactNode;
  flexGrow?: number;
  height?: number;
  width?: number;
  minHeight?: number;
}) {
  return (
    <Box flexDirection="column" paddingX={1} marginBottom={1} flexGrow={flexGrow} height={height} width={width} minHeight={minHeight}>
      <Text color={C.accent} bold>
        {title.toUpperCase()}
      </Text>
      {children}
    </Box>
  );
}
