import json
from typing import Dict, Any, Optional, List

from surogate.core.config.dataset_config import ConversationDatasetConfig
from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.utils.logger import get_logger

logger = get_logger()

class ConversationPreprocessor(RowPreprocessor):
    def __init__(self, dataset_config: ConversationDatasetConfig):
        super().__init__()
        self.ds_cfg = dataset_config
        self.message_property_mappings = dataset_config.message_property_mappings
        self.messages_field = dataset_config.messages_field
        self.completion_field = dataset_config.completion_field
        self.system_field = dataset_config.system_field or "system"
        self.tools_field = dataset_config.tools_field or "tools"
        self.columns[self.messages_field] = "messages"
        if self.completion_field:
            self.columns[self.completion_field] = "completion"

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row["messages"] = self.get_conversation_thread(row)
        tools = self._get_tools(row)
        if tools:
            row["tools"] = tools
        return row

    def get_conversation_thread(self, row):
        turns = []
        messages = self._get_messages(row)
        possible_sys_turn = self.transform_message(messages[0])

        if possible_sys_turn.get("role", None) != "system" and self.system_field in row:
            turn = {"role": "system", "content": row.get(self.system_field)}
            turns.append(turn)

        for message in messages:
            turns.append(self.transform_message(message))

        return turns

    def transform_message(self, message: dict) -> dict:
        transformed_message = {}
        for key, value in self.message_property_mappings.items():
            if message.get(value) is not None:
                transformed_message[key] = message[value]
            else:
                logger.debug(
                    f"Could not find value for property {value} in message: {message}"
                )

        # Map the role if necessary
        if "tool_calls" in transformed_message and transformed_message["tool_calls"]:
            for tool_call in transformed_message["tool_calls"]:
                if "function" in tool_call and "arguments" in tool_call["function"]:
                    args = tool_call["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            tool_call["function"]["arguments"] = json.loads(args)
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Error parsing tool_calls arguments as JSON. "
                                f"Function: {tool_call.get('function', {}).get('name', 'unknown')}, "
                                f"Arguments string: {args!r}, "
                                f"Error: {e}"
                            )
                            raise

        return transformed_message

    def _get_messages(self, row):
        messages = row.get("messages", None)
        if messages is None:
            raise ValueError("Messages is null. Please check `messages_field`.")

        if not isinstance(messages, list):
            raise ValueError(
                "Unknown messages format. Please convert it into a list[dict].\n"
                f"Current format: {type(messages)}"
            )

        if self.completion_field:
            completion = row.get("completion", None)
            if completion is not None:
                if isinstance(completion, list):
                    messages = messages + completion
                elif isinstance(completion, dict):
                    messages = messages + [completion]

        return messages

    def _get_tools(self, row) -> list[dict] | None:
        """Get tools from prompt if available."""
        tools = row.get(self.tools_field, None)
        if tools is None:
            return None

        if isinstance(tools, list):
            return tools

        raise ValueError(
            "Unknown tools format. Please convert it into a list[dict].\n"
            f"Current format: {type(tools)}"
        )

    def preprocess_batch(self, rows: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of conversation dataset rows efficiently.

        1. Batches message extraction and transformation
        2. Batches JSON parsing for tool_calls
        3. Reduces function call overhead

        Args:
            rows: List of input row dictionaries

        Returns:
            List of processed dictionaries with 'messages' and optionally 'tools' fields
        """
        results = []

        for row in rows:
            # Process messages
            messages = self.get_conversation_thread(row)
            result = {"messages": messages}

            # Process tools if present
            tools = self._get_tools(row)
            if tools:
                result["tools"] = tools

            results.append(result)

        return results
