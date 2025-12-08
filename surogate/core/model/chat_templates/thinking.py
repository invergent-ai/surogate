from .base import ChatTemplateProcessor


class ThinkingChatTemplateProcessor(ChatTemplateProcessor):
    with_answer = False
    no_think_prefix = ''  # for hybrid thinking model
    history_think_prefix = ''
    add_no_think_prefix_after_tool = True

class ThinkingWithAnswerChatTemplateCoder(ThinkingChatTemplateProcessor):
    with_answer = True