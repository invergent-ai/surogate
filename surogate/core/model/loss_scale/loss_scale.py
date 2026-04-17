from typing import Literal

from surogate.core.model.utils import ContextType
from surogate.utils.messages import Messages, get_last_user_round

ALL_BASE_STRATEGY = ["default", "last_round", "all"]


class LossScale:
    is_binary = True

    def __init__(self, base_strategy: Literal["default", "last_round", "all"] = "default"):
        assert base_strategy in ALL_BASE_STRATEGY, (
            f"ALL_BASE_STRATEGY: {ALL_BASE_STRATEGY}, base_strategy: {base_strategy}"
        )
        self.base_strategy = base_strategy

    def __call__(
        self, context_list: list[str], context_types: list[ContextType], messages: Messages, **kwargs
    ) -> tuple[list[str], list[float]]:
        res_context_list = []
        res_loss_scale = []
        i = 0
        last_user_round = get_last_user_round(messages)
        for context, context_type in zip(context_list, context_types):
            is_last_round = 2 * i >= last_user_round
            loss = None
            if context_type == ContextType.RESPONSE:
                # Currently, we only support applying loss/mask to the response part.
                loss = messages[2 * i + 1].get("loss")
                assert context == messages[2 * i + 1]["content"]
                i += 1
            if isinstance(context, dict) and "loss_scale" in context:
                res_context_list += [[token] for token in context["token_ids"]]
                res_loss_scale += context["loss_scale"]
            else:
                if isinstance(context, dict) and "token_ids" in context:
                    context = context["token_ids"]
                if context_type == ContextType.RESPONSE and loss is not None:
                    res_context_list.append(context)
                    res_loss_scale.append(float(loss))
                else:
                    is_assistant = context_type in {ContextType.RESPONSE, ContextType.SUFFIX}
                    if (
                        self.base_strategy == "all"
                        or (self.base_strategy == "default" and is_assistant)
                        or (self.base_strategy == "last_round" and is_assistant and is_last_round)
                    ):
                        res_context_list.append(context)
                        res_loss_scale.append(1.0)
                    else:
                        res_context_list.append(context)
                        res_loss_scale.append(0.0)
        return res_context_list, res_loss_scale

    @property
    def is_loss_scale_binary(self):
        return self.is_binary


loss_scale_map = {
    "-": LossScale,
    "default": LossScale,
    "last_round": LossScale,
    "all": LossScale,
}

for k, v in loss_scale_map.items():
    v.name = k


def get_loss_scale(loss_scale: str) -> LossScale:
    splited = loss_scale.split("+", 1)
    if len(splited) == 1:
        if splited[0] in ALL_BASE_STRATEGY:
            base_strategy, loss_scale = splited[0], "-"
        else:
            base_strategy, loss_scale = "default", splited[0]
    else:
        base_strategy, loss_scale = splited
    if loss_scale not in loss_scale_map:
        raise ValueError(f"Unknown loss_scale: {loss_scale!r}. Supported: {list(loss_scale_map.keys())}")
    return loss_scale_map[loss_scale](base_strategy)
