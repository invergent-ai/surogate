import json
import os
from typing import Tuple, List, Optional, Dict, Literal

from surogate.core.model.chat_templates.utils import split_str_parts_by, Messages, get_last_user_round
from surogate.core.model.utils import ContextType
from surogate.utils.logger import get_logger

logger = get_logger()

ALL_BASE_STRATEGY = ['default', 'last_round', 'all']

class LossScale:
    # Indicates whether loss_scale contains only 0 and 1.
    # If set to True, loss_scale will be replaced by labels to stay compatible with
    # acceleration techniques such as liger_kernel.
    # If set to False, an additional 'loss_scale' key will be stored and the
    # corresponding loss function will be used.
    loss_scale_config = None  # path
    is_binary = None

    def __init__(self, base_strategy: Literal['default', 'last_round', 'all'] = 'default'):
        assert base_strategy in ALL_BASE_STRATEGY, (
            f'ALL_BASE_STRATEGY: {ALL_BASE_STRATEGY}, base_strategy: {base_strategy}')
        self.base_strategy = base_strategy
        self.loss_scale_map = None
        if self.loss_scale_config is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, 'loss_scales', self.loss_scale_config)
            with open(config_path, 'r', encoding='utf-8') as json_file:
                self.loss_scale_map = json.load(json_file)

    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        """Calculate loss scale

        Args:
            context: The input context
            query: The query of this round.

        Returns:
            A tuple, list of context and list of loss_scales
        """
        return [context], [1.]

    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,
                 **kwargs) -> Tuple[List[str], List[float]]:
        res_context_list = []
        res_loss_scale = []
        i = 0
        last_user_round = get_last_user_round(messages)
        for context, context_type in zip(context_list, context_types):
            is_last_round = 2 * i >= last_user_round
            query, loss = None, None
            if context_type == ContextType.RESPONSE:
                query = messages[2 * i]['content']
                # Currently, we only support applying loss/mask to the response part.
                loss = messages[2 * i + 1].get('loss')
                assert context == messages[2 * i + 1]['content']
                i += 1
            if isinstance(context, dict) and 'loss_scale' in context:
                new_context = [[token] for token in context['token_ids']]
                loss_scale = context['loss_scale']
            else:
                if isinstance(context, dict) and 'token_ids' in context:
                    context = context['token_ids']
                if context_type == ContextType.RESPONSE and loss is not None:
                    new_context, loss_scale = [context], [float(loss)]
                else:
                    is_assistant = context_type in {ContextType.RESPONSE, ContextType.SUFFIX}
                    if self.base_strategy == 'all' or (self.base_strategy == 'default'
                                                       and is_assistant) or (self.base_strategy == 'last_round'
                                                                             and is_assistant and is_last_round):
                        new_context, loss_scale = self.get_loss_scale(context, query=query)
                    else:
                        new_context, loss_scale = [context], [0.]
            res_context_list += new_context
            res_loss_scale += loss_scale
        return res_context_list, res_loss_scale

    @property
    def is_loss_scale_binary(self):
        if self.is_binary is not None:
            return self.is_binary
        if self.loss_scale_map is None:
            return True
        return all(scale == 0.0 or scale == 1.0 for lst in self.loss_scale_map.values() for scale in lst)


class DefaultLossScale(LossScale):
    pass


class LastRoundLossScale(LossScale):
    def get_loss_scale(self, context: str, context_type: ContextType, is_last_round: bool, **kwargs):
        if context_type == ContextType.RESPONSE:
            return [context], [float(is_last_round)]
        return super().get_loss_scale(context, context_type, is_last_round)


class TrainAllLossScale(LossScale):
    def get_loss_scale(self, context: str, context_type: ContextType, *args, **kwargs):
        return [context], [1.]


class REACTLossScale(LossScale):
    loss_scale_config = 'react.json'

    def get_loss_scale(self,
                       context: str,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None):
        if context_type == ContextType.RESPONSE and isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context, context_type, is_last_round)

class IgnoreEmptyThink(REACTLossScale):
    loss_scale_config = 'ignore_empty_think.json'


class LastRoundWithIgnoreEmptyThink(LossScale):
    loss_scale_config = 'ignore_empty_think.json'

class HermesLossScale(REACTLossScale):
    loss_scale_config = 'hermes.json'

class QwenLossScale(REACTLossScale):
    loss_scale_config = 'qwen.json'

def calculate_loss_scale(
        query: str,
        response: str,
        response_loss_scale_map: Dict[str, list],
        query_loss_scale_map: Optional[Dict[str, list]] = None
) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.

    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf

    Agent response format:

    ```text
        Thought: you should always think about what to do
        Action: the action to take, should be one of the above tools[fire_recognition,
            fire_alert, call_police, call_fireman]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ```
    Returns:
        A tuple of agent response parts and their weights.
    """
    # query loss scale map
    if query_loss_scale_map is not None:
        for key in query_loss_scale_map.keys():
            if key in query:
                if isinstance(query_loss_scale_map[key], (float, int)):
                    query_loss_scale_map[key] = [query_loss_scale_map[key]]
                loss_scale_value = query_loss_scale_map[key][0]
                return [response], [float(loss_scale_value)]
    delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 2]
    if delimiters:
        agent_parts = split_str_parts_by(response, delimiters)
    else:
        regex_delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 1]
        agent_parts = split_str_parts_by(response, regex_delimiters, regex_mode=True)
    weights = []
    agent_content = []
    for c in agent_parts:
        if c['key'] in response_loss_scale_map:
            loss_scale = response_loss_scale_map[c['key']]
            assert len(loss_scale) in {1, 2}, f'loss_scale: {loss_scale}'
            if len(loss_scale) == 1:
                weights += loss_scale
                agent_content.append(c['content'])
            else:
                weights += loss_scale
                agent_content += [c['key'], c['content']]
        else:
            weights.append(1.)
            agent_content.append(c['content'])
    return agent_content, weights


loss_scale_map = {
    '-': LossScale,
    'last_round': LastRoundLossScale,
    'default': DefaultLossScale,
    'all': TrainAllLossScale,
    'ignore_empty_think': IgnoreEmptyThink,
    'last_round_with_ignore_empty_think': LastRoundWithIgnoreEmptyThink,
    # agent
    'react': REACTLossScale,
    'hermes': HermesLossScale,
    'qwen': QwenLossScale,
}

for k, v in loss_scale_map.items():
    v.name = k
    
    
def get_loss_scale(loss_scale: str) -> LossScale:
    splited = loss_scale.split('+', 1)
    if len(splited) == 1:
        if splited[0] in ALL_BASE_STRATEGY:
            base_strategy, loss_scale = splited[0], '-'
        else:
            base_strategy, loss_scale = 'default', splited[0]
    else:
        base_strategy, loss_scale = splited
    return loss_scale_map[loss_scale](base_strategy)