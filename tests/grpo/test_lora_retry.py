"""Which failures of the weight-update POST are worth another attempt.

`load_lora_adapter` already wraps its POST in ten attempts with exponential
backoff, but the predicate deciding what to retry only ever looked at HTTP
statuses. A connection that dies without producing a status fell straight
through, so a single dropped socket ended a run that had already trained and
already written its adapter, with a non-zero exit and nothing lost but the
exit code.

That is not hypothetical: the engine's HTTP pool used to close connections past
its backlog with no response at all, and a final eval fanning out wider than the
backlog took the weight update with it. The pool is fixed, and this stays,
because the request is idempotent and any transport error on it is worth another
attempt whatever caused it.
"""

from __future__ import annotations

import httpx

from surogate.grpo.utils.client import _is_retryable_lora_error


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://engine/load_lora_adapter")
    return httpx.HTTPStatusError(
        f"{code}", request=request, response=httpx.Response(code, request=request)
    )


def test_a_connection_that_dies_without_a_status_is_retried():
    """The case that ended a finished run: a socket closed with no response."""
    request = httpx.Request("POST", "http://engine/load_lora_adapter")
    for error in (
        httpx.ReadError("peer closed", request=request),
        httpx.ConnectError("refused", request=request),
        httpx.RemoteProtocolError("incomplete", request=request),
        httpx.ConnectTimeout("no answer", request=request),
    ):
        assert _is_retryable_lora_error(error), f"{type(error).__name__} should be retried"


def test_a_read_timeout_is_not_retried_even_though_it_is_a_transport_error():
    """The server has the request and is probably still reloading.

    `_apply_policy_update` holds all new rollout scheduling while this runs and
    leaves `checkpoint_ready` cleared, so each further 600s wait is a run that
    has silently stopped making progress. Ten of them is the multi-hour wedge
    the deadline exists to end. The reload may well finish on its own; the next
    weight update will load it.
    """
    request = httpx.Request("POST", "http://engine/load_lora_adapter")
    assert not _is_retryable_lora_error(httpx.ReadTimeout("no answer yet", request=request))


def test_the_statuses_that_were_already_retried_still_are():
    """404 is the adapter not being visible yet, 500 a reload still settling."""
    assert _is_retryable_lora_error(_status_error(404))
    assert _is_retryable_lora_error(_status_error(500))


def test_a_rejection_the_server_meant_is_not_retried():
    """A 400 says the request is wrong; sending it ten times keeps it wrong."""
    assert not _is_retryable_lora_error(_status_error(400))
    assert not _is_retryable_lora_error(_status_error(403))


def test_an_unrelated_exception_is_not_retried():
    assert not _is_retryable_lora_error(ValueError("not an httpx failure"))
