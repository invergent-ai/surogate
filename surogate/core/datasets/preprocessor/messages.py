import ast
from typing import Any, Callable, Dict, List, Optional, Union

from surogate.core.datasets.preprocessor.row import RowPreprocessor


def default_repair_messages(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s

class MessagesPreprocessor(RowPreprocessor):

    def __init__(
            self,
            *,
            # If set to None, automatic matching will be performed.
            role_key: Optional[str] = None,  # 'role', 'from'
            content_key: Optional[str] = None,  # 'content', 'value'
            user_role: Optional[str] = None,  # 'user', 'human'
            assistant_role: Optional[str] = None,  # 'assistant', 'gpt', 'bot'
            system_role: str = 'system',
            # 'conversation', 'conversations' -> 'messages'
            columns: Optional[Dict[str, str]] = None,
            repair_messages: Callable[[Union[str, List[Dict[str, str]]]],
            Optional[List[Dict[str, str]]]] = default_repair_messages,
            inner_key: Optional[str] = None,
            **kwargs):
        super().__init__(columns=columns, **kwargs)
        self.role_keys = ['role', 'from'] if role_key is None else [role_key]
        self.content_keys = ['content', 'value'] if content_key is None else [content_key]
        self.user_roles = ['user', 'human'] if user_role is None else [user_role]
        self.assistant_roles = ['assistant', 'gpt', 'bot'] if assistant_role is None else [assistant_role]
        self.tool_call_roles = ['function_call']
        self.tool_response_roles = ['function_response', 'observation', 'observations']

        self.system_role = system_role
        self.repair_messages = repair_messages
        self.inner_key = inner_key

        message_keys = ['messages', 'conversation', 'conversations']
        for key in message_keys:
            self.columns[key] = 'messages'
        # sharegptq
        system_keys = ['system', 'system_prompt']
        if system_role not in system_keys:
            system_keys.append(system_role)
        for key in system_keys:
            self.columns[key] = 'system'

    @staticmethod
    def _is_sharegpt_format(message: Dict[str, str]) -> bool:
        if 'role' in message or 'content' in message:
            return False
        return True

    def sharegpt_to_messages(self, messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
        self._to_std_key(messages, 'user', self.user_roles)
        self._to_std_key(messages, 'assistant', self.assistant_roles)
        new_messages = []
        if system is not None:
            new_messages.append({'role': 'system', 'content': system})
        for message in messages:
            user_message = {'role': 'user', 'content': message['user']}
            assistant_message = {'role': 'assistant', 'content': message['assistant']}
            new_messages.append(user_message)
            new_messages.append(assistant_message)
        return new_messages

    def to_std_messages(self, messages: List[Dict[str, str]], system: Optional[str]) -> None:
        if messages[0]['role'] == self.system_role:
            messages[0]['role'] = 'system'
        elif system is not None:
            messages.insert(0, {'role': 'system', 'content': system})
        for message in messages:
            role = message['role']
            if role in self.user_roles:
                message['role'] = 'user'
            elif role in self.assistant_roles:
                message['role'] = 'assistant'
            elif role.replace('-', '_') in self.tool_call_roles:
                message['role'] = 'tool_call'
            elif role.replace('-', '_') in self.tool_response_roles:
                message['role'] = 'tool_response'

    @staticmethod
    def _to_std_key(messages: List[Dict[str, str]], std_key: str, optional_keys: List[str]) -> None:
        for message in messages:
            for key in optional_keys:
                if key in message:
                    message[std_key] = message.pop(key)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if 'rejected_messages' in row:
            row['rejected_messages'] = MessagesPreprocessor.preprocess(
                self, {'messages': row['rejected_messages']})['messages']
        messages = row['messages']
        if self.inner_key is not None:
            messages = messages[self.inner_key]
        messages: Optional[List[Dict[str, str]]] = self.repair_messages(messages)
        if not messages or isinstance(messages, str):
            return
        self._to_std_key(messages, 'role', self.role_keys)
        self._to_std_key(messages, 'content', self.content_keys)
        system = row.get('system', None)
        if self._is_sharegpt_format(messages[0]):
            messages = self.sharegpt_to_messages(messages, system)
        else:
            self.to_std_messages(messages, system)  # inplace
        row['messages'] = messages
        return row

    def preprocess_batch(self, rows: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of messages dataset rows efficiently.

        1. Batches message repair and transformation operations
        2. Reduces function call overhead
        3. Batches format detection and conversion

        Args:
            rows: List of input row dictionaries

        Returns:
            List of processed dictionaries with standardized 'messages' field
        """
        results = []

        for row in rows:
            # Handle rejected messages if present
            if 'rejected_messages' in row:
                row['rejected_messages'] = MessagesPreprocessor.preprocess(
                    self, {'messages': row['rejected_messages']})['messages']

            messages = row['messages']
            if self.inner_key is not None:
                messages = messages[self.inner_key]

            messages: Optional[List[Dict[str, str]]] = self.repair_messages(messages)
            if not messages or isinstance(messages, str):
                results.append(None)
                continue

            self._to_std_key(messages, 'role', self.role_keys)
            self._to_std_key(messages, 'content', self.content_keys)
            system = row.get('system', None)

            if self._is_sharegpt_format(messages[0]):
                messages = self.sharegpt_to_messages(messages, system)
            else:
                self.to_std_messages(messages, system)  # inplace

            row['messages'] = messages
            results.append(row)

        return results