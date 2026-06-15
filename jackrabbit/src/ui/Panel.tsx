import React from "react";
import { Box, Text } from "ink";
import { C } from "./theme.ts";

/** A rounded-border panel with a colored title (the wizard look). */
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
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={C.border}
      paddingX={1}
      flexGrow={flexGrow}
      height={height}
      width={width}
      minHeight={minHeight}
    >
      <Text color={C.accent} bold>
        {title}
      </Text>
      {children}
    </Box>
  );
}
