"""Why a GRPO watchdog aborted, when it has no way to say so by raising.

Both GRPO runners stop the pipeline the same way: a watchdog thread signals its
own process, and the main thread -- blocked in ``asyncio.run`` -- receives a
``KeyboardInterrupt`` indistinguishable from the one a user's Ctrl-C produces.
With nowhere to record the cause, the two cases could not be told apart, so a
crashed component exited 0 and the run was finalized as ``completed`` with no
error against it.

The convention both runners follow:

1. The watchdog records the reason **before** signalling, so it is already
   visible by the time the interrupt lands.
2. The main thread raises only **after** its teardown block has reaped the vLLM
   process trees. Exiting early would strand them holding their GPUs.
"""


class AbortReason:
    """A one-slot box shared between the watchdog thread and the main thread.

    Holds the reason a watchdog aborted the pipeline, or ``None`` if it did
    not. It has to be a mutable object rather than a plain string: the value is
    written by the watchdog thread and read by the main thread.
    """

    def __init__(self, reason: str | None = None):
        self.reason = reason
