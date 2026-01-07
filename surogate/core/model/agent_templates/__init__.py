from typing import Dict

from surogate.core.model.agent_templates.base import BaseAgentTemplate
from surogate.core.model.agent_templates.llama import Llama3AgentTemplate
from surogate.core.model.agent_templates.react import ReactAgentTemplate
from surogate.core.model.agent_templates.hermes import HermesAgentTemplate

agent_templates: Dict[str, type[BaseAgentTemplate]] = {
    # ref: https://qwen.readthedocs.io/zh-cn/latest/framework/function_call.html#function-calling-templates
    'react': ReactAgentTemplate,
    'llama3': Llama3AgentTemplate,
    'hermes': HermesAgentTemplate,
}