import pytest

from ultra.schemas import Workflow, WorkflowStep
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
    validate_workflow(wf, worker_count=1)
    with pytest.raises(WorkflowValidationError):
        parse_workflow("{not json")
