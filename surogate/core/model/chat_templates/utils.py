import re
from typing import Dict, Any, Union, Tuple, List, Set, Optional, Type

import torch

Tool = Dict[str, Union[str, Dict]]
History = List[Union[Tuple[str, str], List[str]]]
Message = Dict[str, Union[str, List[Dict[str, Any]], List[int], None]]
Messages = List[Message]

def history_to_messages(history: History,
                        system: Optional[str] = None,
                        roles: Optional[List[List[str]]] = None) -> 'Messages':
    """
    history: [['query1', 'response1'], ['query2', 'response2']]
        or [['query1', 'response1'], ['query2', None]]
    """
    messages = []
    if not roles:
        roles = [['user', 'assistant']] * len(history)
    else:
        assert len(roles) == len(history), f'len(roles): {len(roles)}, len(history): {len(history)}'
    if system is not None:
        messages.append({'role': 'system', 'content': system})

    for role, h in zip(roles, history):
        assert isinstance(h, (list, tuple))
        if h[0] is not None:
            messages.append({'role': role[0], 'content': h[0]})
        if h[1] is not None:
            messages.append({'role': role[1], 'content': h[1]})
    return messages

def messages_to_history(messages: 'Messages') -> Dict[str, Any]:
    system = None
    messages = messages.copy()
    if messages[0]['role'] == 'system':
        system = messages[0]['content']
        messages = messages[1::]
    if len(messages) % 2 == 1:
        messages.append({'role': 'assistant', 'content': None})
    history = []
    history_roles = []
    for user_message, assistant_message in zip(messages[::2], messages[1::2]):
        assert user_message['role'] in {'tool', 'user'}, f'user_message {user_message}'
        assert assistant_message['role'] == 'assistant', f'assistant_message: {assistant_message}'
        history.append([user_message['content'], assistant_message['content']])
        history_roles.append([user_message['role'], assistant_message['role']])
    query, response = history.pop() if history else (None, None)
    query_role = history_roles.pop()[0] if history_roles else None
    return {
        'history': history,
        'history_roles': history_roles,
        'query': query,
        'query_role': query_role,
        'response': response,
        'system': system,
    }


def fetch_one(element: Union[Tuple, List, Set, Dict, Any], item_type: Optional[Type] = None) -> Any:
    if isinstance(element, (tuple, set, list)):
        for ele in element:
            out = fetch_one(ele)
            if out and (item_type is None or isinstance(out, item_type)):
                return out
    elif isinstance(element, dict):
        return fetch_one(list(element.values()))
    else:
        return element


def _split_str_by_regex(text: str, regex_delimiters: List[str]) -> List[str]:
    combined_pattern = '|'.join(f'({pattern})' for pattern in regex_delimiters)
    parts = re.split(combined_pattern, text, flags=re.DOTALL)
    parts = [part for part in parts if part is not None]
    if parts[0] == '':
        parts.pop(0)
    else:
        parts.insert(0, '')
    assert len(parts) % 2 == 0, f'result: {parts}'
    assert ''.join(parts) == text, f'split_result: {parts}, text: {text}'
    return parts


def split_str_parts_by(text: str, delimiters: List[str], regex_mode: bool = False) -> List[Dict[str, str]]:
    """Split the text field into parts.

    Args:
        text: A text to be split.
        delimiters: The delimiters.

    Returns:
        The split text in list of dicts.
    """
    assert isinstance(text, str), f'text: {text}'
    delimiters_origin = delimiters
    if not regex_mode:
        delimiters = [re.escape(delimiter) for delimiter in delimiters]
    parts = _split_str_by_regex(text, delimiters) if delimiters else ['', text]
    res = []
    if regex_mode:
        parts = [part for part in parts if part]
        for part in parts:
            for delimiter, delimiter_origin in zip(delimiters, delimiters_origin):
                if re.match(delimiter, part, re.DOTALL):
                    break
            else:
                delimiter_origin = ''
            res.append({'key': delimiter_origin, 'content': part})
    else:
        for key, content in zip(parts[::2], parts[1::2]):
            res.append({'key': key, 'content': content})
    return res

def get_last_user_round(messages):
    """Get the index of the last occurrence of user role"""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] == 'user':
            return i
    return -1

def findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """Find the index of a token in the token_list."""
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(sub_token_list[0], idx + 1)
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx:idx + len(sub_token_list)]:
                res.append(idx)
    except ValueError:
        pass
    return res


def get_packed_seq_params(position_ids: torch.Tensor):
    assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
    position_ids_f = position_ids.flatten()
    indices_q = torch.arange(position_ids_f.shape[0], device=position_ids_f.device, dtype=torch.int32)

    if position_ids_f.numel() == 0:
        cu_seqlens = torch.tensor([0], device=position_ids_f.device, dtype=torch.int32)
    else:
        diffs = position_ids_f[1:] - position_ids_f[:-1]
        boundaries = torch.where(diffs != 1)[0] + 1
        starts = torch.cat([
            torch.tensor([0], device=position_ids_f.device, dtype=boundaries.dtype),
            boundaries,
        ])
        cu_seqlens = torch.cat([
            starts,
            torch.tensor(position_ids_f.shape, device=position_ids_f.device, dtype=torch.int32),
        ])

    max_length = cu_seqlens.diff().max()  # position_ids_f.max() + 1
    return {
        'cu_seq_lens_q': cu_seqlens,
        'cu_seq_lens_k': cu_seqlens,
        'max_length_q': max_length,
        'max_length_k': max_length,
    }
