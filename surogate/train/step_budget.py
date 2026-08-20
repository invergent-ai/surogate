"""Guard against a run that cannot take a single optimizer step.

The step budget is computed in *tokens*, not rows::

    tokens_per_step = per_device_train_batch_size * sequence_len * gpus
                      * gradient_accumulation_steps
    steps_per_epoch = dataset_tokens // tokens_per_step

so a dataset smaller than one step's worth of tokens yields ``0`` and the
training loop runs zero times. Left unchecked the run then logs
"Training loop completed successfully after step -1", writes out an
adapter that never saw a gradient, and reports success — indistinguishable
from a real run to anyone reading the UI.
"""

from __future__ import annotations


class ZeroStepBudgetError(ValueError):
    """The configured budget yields no optimizer steps."""


def check_step_budget(
    max_steps: int,
    *,
    dataset_tokens: int | None = None,
    tokens_per_step: int | None = None,
    batch_size: int | None = None,
    sequence_len: int | None = None,
    gpus: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> None:
    """Raise if *max_steps* is not positive.

    Everything after ``max_steps`` is optional and only shapes the message:
    the point of the error is to say which knob to turn, since "0 steps" on
    its own sends people looking at their data when the batch settings are
    usually what's wrong.
    """
    if max_steps > 0:
        return

    detail = ""
    if dataset_tokens is not None and tokens_per_step:
        shortfall = tokens_per_step - dataset_tokens
        detail = (
            f" The dataset holds {dataset_tokens:,} tokens but one step "
            f"consumes {tokens_per_step:,}"
        )
        parts = [
            f"per_device_train_batch_size={batch_size}" if batch_size else "",
            f"sequence_len={sequence_len}" if sequence_len else "",
            f"gpus={gpus}" if gpus else "",
            (
                f"gradient_accumulation_steps={gradient_accumulation_steps}"
                if gradient_accumulation_steps else ""
            ),
        ]
        parts = [p for p in parts if p]
        if parts:
            detail += " (" + " x ".join(parts) + ")"
        detail += f", short by {shortfall:,}."

    raise ZeroStepBudgetError(
        "This configuration trains for 0 optimizer steps, so the run would "
        "produce an untrained model." + detail +
        " Use a larger dataset, a shorter sequence_len, or a smaller "
        "effective batch (batch size x gradient accumulation). Set "
        "max_steps explicitly to override."
    )
