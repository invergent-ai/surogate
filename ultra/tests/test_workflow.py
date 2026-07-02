import pytest

from ultra.schemas import Workflow, WorkflowStep
from ultra.harness.base import wall_time_cap_seconds
from ultra.workflow import WorkflowValidationError, parse_workflow, validate_workflow


def _wf(steps):
    return Workflow(steps=[WorkflowStep(**s) for s in steps])


def test_valid_two_step():
    wf = _wf([
        {"worker_id": 0, "subtask": "plan", "access": []},
        {"worker_id": 1, "subtask": "solve", "access": [0]},
    ])
    validate_workflow(wf, worker_count=2)  # must not raise


def test_rejects_empty_and_too_many():
    with pytest.raises(WorkflowValidationError):
        validate_workflow(Workflow(steps=[]), worker_count=2)
    six = _wf([{"worker_id": 0, "subtask": "s", "access": []} for _ in range(6)])
    with pytest.raises(WorkflowValidationError):
        validate_workflow(six, worker_count=2)


def test_rejects_bad_worker_id():
    with pytest.raises(WorkflowValidationError):
        validate_workflow(_wf([{"worker_id": 5, "subtask": "s", "access": []}]), worker_count=2)


def test_rejects_empty_subtask():
    with pytest.raises(WorkflowValidationError):
        validate_workflow(_wf([{"worker_id": 0, "subtask": "   ", "access": []}]), worker_count=2)


def test_rejects_forward_and_self_access():
    forward = _wf([
        {"worker_id": 0, "subtask": "s", "access": [1]},
        {"worker_id": 0, "subtask": "t", "access": []},
    ])
    with pytest.raises(WorkflowValidationError):
        validate_workflow(forward, worker_count=2)
    with pytest.raises(WorkflowValidationError):
        validate_workflow(_wf([{"worker_id": 0, "subtask": "s", "access": [0]}]), worker_count=2)


def test_rejects_duplicate_access():
    wf = _wf([
        {"worker_id": 0, "subtask": "a", "access": []},
        {"worker_id": 0, "subtask": "b", "access": []},
        {"worker_id": 0, "subtask": "c", "access": [0, 0]},
    ])
    with pytest.raises(WorkflowValidationError):
        validate_workflow(wf, worker_count=2)


def test_parse_valid_and_invalid_json():
    wf = parse_workflow('{"steps":[{"worker_id":0,"subtask":"go","access":[]}]}')
    assert len(wf.steps) == 1
    assert wf.steps[0].budget == "medium"
    validate_workflow(wf, worker_count=1)
    with pytest.raises(WorkflowValidationError):
        parse_workflow("{not json")


def test_parse_tolerates_escaped_apostrophe_only():
    wf = parse_workflow('{"steps":[{"worker_id":0,"subtask":"worker\\\'s fix","access":[]}]}')

    assert wf.steps[0].subtask == "worker's fix"


def test_budget_wall_time_caps_take_strictest_finite_limit():
    assert wall_time_cap_seconds("short", task_cap=1800, harness_cap=900) == 900
    assert wall_time_cap_seconds("short", task_cap=1800, harness_cap=2000) == 1200
    assert wall_time_cap_seconds("max", task_cap=1800, harness_cap=None) == 1800
    assert wall_time_cap_seconds("max", task_cap=None, harness_cap=None) is None
