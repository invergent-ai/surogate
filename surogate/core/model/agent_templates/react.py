from typing import List, Union, Optional

from surogate.core.model.agent_templates.base import BaseAgentTemplate


class ReactAgentTemplate(BaseAgentTemplate):

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_names = []
        tool_descs = []
        for tool in tools:
            tool_desc = self._parse_tool(tool)
            tool_names.append(tool_desc.name_for_model)
            tool_descs.append(
                f'{tool_desc.name_for_model}: Call this tool to interact with the {tool_desc.name_for_human} API. '
                f'What is the {tool_desc.name_for_human} API useful for? {tool_desc.description_for_model} '
                f'Parameters: {tool_desc.parameters} {tool_desc.args_format}')

        return """Answer the following questions as best you can. You have access to the following tools:

""" + '\n\n'.join(tool_descs) + f"""

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{','.join(tool_names)}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""