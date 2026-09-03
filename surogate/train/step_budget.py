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
    """Raise if the run cannot take a full optimizer step.

    Two ways that happens, and both must be caught here:

    * the budget resolved to zero, which is what an epochs-based run yields
      when ``dataset_tokens // tokens_per_step`` floors to 0; and
    * the dataset cannot fill a single step even though ``max_steps`` was
      given explicitly. A typed-in number is not evidence the data exists,
      and left unchecked the run dies later in the dataloader with
      ``No more files to load``, which names neither the dataset nor the
      batch settings.

    Note this is deliberately *not* "fewer steps than requested". Asking for
    more steps than one epoch holds is normal, the loader wraps. Only a
    dataset too small for one step is a defect.

    Everything after ``max_steps`` is optional and only shapes the message:
    the point of the error is to say which knob to turn, since "0 steps" on
    its own sends people looking at their data when the batch settings are
    usually what's wrong.
    """
    starved = (
        dataset_tokens is not None
        and tokens_per_step
        and dataset_tokens < tokens_per_step
    )
    if max_steps > 0 and not starved:
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

    # Two different situations, and telling them apart matters: one is a
    # budget that resolved to nothing, the other is a budget that was asked
    # for and cannot be met.
    if starved and max_steps > 0:
        headline = (
            f"The dataset cannot fill a single optimizer step, so the "
            f"requested {max_steps} step(s) cannot run."
        )
    else:
        headline = (
            "This configuration trains for 0 optimizer steps, so the run "
            "would produce an untrained model."
        )

    # Deliberately no longer suggests setting max_steps. That never let
    # anyone train: it only swapped this message for the dataloader's
    # `No more files to load`, since the data is still absent either way.
    raise ZeroStepBudgetError(
        headline + detail +
        " Use a larger dataset, a shorter sequence_len, or a smaller "
        "effective batch (batch size x gradient accumulation)."
    )
