"""Expert-designed disagreement tasks for Fugu-Ultra.

These tasks are deliberately small, verifier-backed, and text-only. The point is
not source volume; it is to create controlled cases where workers are likely to
disagree because the task requires precedence handling, exact implementation
details, multi-step tool use, or rejection of stale context.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from .schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)
from .tool_dialog_tasks import COMMON_TOOLS

SOURCE_NAME = "expert_disagreement_v1"
SOURCE_VERSION = "v1"
SOURCE_NAME_V2 = "expert_disagreement_v2"
SOURCE_VERSION_V2 = "v2"


@dataclass(frozen=True)
class DirectExpertTask:
    task_id: str
    domain: str
    prompt: str
    grader_type: str
    answer: Any
    mechanism: str


@dataclass(frozen=True)
class CodeExpertTask:
    task_id: str
    prompt: str
    entry_point: str
    test: str
    mechanism: str


@dataclass(frozen=True)
class ToolExpertTask:
    task_id: str
    domain: str
    instruction: str
    initial_state: dict[str, Any]
    success: list[dict[str, Any]]
    allowed_tools: list[str]
    mechanism: str
    max_turns: int = 7


@dataclass(frozen=True)
class LongExpertTask:
    task_id: str
    prompt: str
    documents: list[dict[str, str]]
    must_contain: list[str]
    must_not_contain: list[str]
    mechanism: str


DIRECT_TASKS: tuple[DirectExpertTask, ...] = (
    DirectExpertTask(
        task_id="precedence-contract-renewal",
        domain="general",
        mechanism="policy_precedence_over_stale_fact",
        grader_type="mc_letter",
        answer="C",
        prompt=(
            "A contract packet uses this precedence: signed amendment > signed memo > draft > FAQ. "
            "The FAQ says renewal is Nov 3. A signed memo says Nov 9. A draft amendment says Nov 14. "
            "A signed amendment says Nov 11 but only for the analytics addendum, not the base contract. "
            "For the base observability contract, which date controls?\n\n"
            "A. Nov 3\nB. Nov 14\nC. Nov 9\nD. Nov 11\n\nReturn only the letter."
        ),
    ),
    DirectExpertTask(
        task_id="negation-count-access-list",
        domain="general",
        mechanism="nested_negation_and_counting",
        grader_type="mc_letter",
        answer="B",
        prompt=(
            "Access is allowed for employees who are not contractors unless they are in the emergency roster. "
            "Mira is an employee, not a contractor. Theo is an employee and contractor, not on the roster. "
            "Nadia is a contractor on the emergency roster. Owen is not an employee. How many are allowed?\n\n"
            "A. 1\nB. 2\nC. 3\nD. 4\n\nReturn only the letter."
        ),
    ),
    DirectExpertTask(
        task_id="conditional-probability-filter",
        domain="math",
        mechanism="base_rate_conditioning",
        grader_type="math_equal",
        answer="4/7",
        prompt=(
            "A filter keeps tickets that are both urgent and signed. There are 12 urgent tickets. "
            "Eight urgent tickets are signed. Six non-urgent tickets are signed. A signed ticket is sampled. "
            "What is P(ticket is urgent | ticket is signed)? Return a reduced fraction."
        ),
    ),
    DirectExpertTask(
        task_id="causal-update-no-leakage",
        domain="science",
        mechanism="causal_vs_correlational_update",
        grader_type="mc_letter",
        answer="D",
        prompt=(
            "A sensor alarm and valve failure are correlated because heat causes both. A technician manually "
            "forces the alarm on while heat is absent. Which inference is justified?\n\n"
            "A. Valve failure becomes very likely because alarm is on.\n"
            "B. Heat becomes very likely because alarm is on.\n"
            "C. Valve failure is impossible.\n"
            "D. The intervention on the alarm alone is not evidence of heat or valve failure.\n\n"
            "Return only the letter."
        ),
    ),
)


CODE_TASKS: tuple[CodeExpertTask, ...] = (
    CodeExpertTask(
        task_id="duration-parser-no-substring-units",
        mechanism="unit_precedence_and_substring_trap",
        entry_point="parse_duration_ms",
        prompt=(
            "Write a Python function parse_duration_ms(text) that parses a duration string and returns "
            "milliseconds. Accepted units are ms, s, min, and h. Units may be separated by spaces. "
            "Do not treat 'min' as containing 'm'. Reject unknown trailing text by raising ValueError."
        ),
        test=r'''
def check(fn):
    assert fn("250ms") == 250
    assert fn("2s 250ms") == 2250
    assert fn("3min 2s") == 182000
    assert fn("1h 1min 1s 1ms") == 3661001
    try:
        fn("1m")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown unit m must fail")
    try:
        fn("1s later")
    except ValueError:
        pass
    else:
        raise AssertionError("trailing text must fail")
''',
    ),
    CodeExpertTask(
        task_id="merge-half-open-intervals",
        mechanism="closed_vs_half_open_boundary",
        entry_point="merge_half_open",
        prompt=(
            "Write merge_half_open(intervals) for half-open integer intervals [start, end). "
            "Merge overlapping intervals only. Adjacent intervals such as [1,3) and [3,5) must stay separate. "
            "Drop empty intervals where start >= end. Return sorted tuples."
        ),
        test=r'''
def check(fn):
    assert fn([(1, 3), (3, 5), (2, 4)]) == [(1, 5)]
    assert fn([(1, 3), (3, 5)]) == [(1, 3), (3, 5)]
    assert fn([(5, 5), (6, 4), (0, 1)]) == [(0, 1)]
    assert fn([(10, 12), (1, 2), (2, 3), (1, 4)]) == [(1, 4), (10, 12)]
''',
    ),
    CodeExpertTask(
        task_id="stable-dedupe-last-wins",
        mechanism="stable_order_with_last_write_wins",
        entry_point="dedupe_last_wins",
        prompt=(
            "Write dedupe_last_wins(records). Each record is a dict with id and value. If an id appears "
            "multiple times, keep its last value, but order ids by their first appearance. Return a list of "
            "(id, value) tuples."
        ),
        test=r'''
def check(fn):
    rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2}, {"id": "a", "value": 3}]
    assert fn(rows) == [("a", 3), ("b", 2)]
    rows = [{"id": "x", "value": 1}, {"id": "y", "value": 2}, {"id": "x", "value": 4}, {"id": "y", "value": 5}]
    assert fn(rows) == [("x", 4), ("y", 5)]
    assert fn([]) == []
''',
    ),
    CodeExpertTask(
        task_id="ledger-balance-with-voids",
        mechanism="state_reconstruction_ignore_voids",
        entry_point="current_balances",
        prompt=(
            "Write current_balances(events). Each event has account, delta, and optional void boolean. "
            "Apply non-void events in order and ignore void events. Return a dict of nonzero balances only."
        ),
        test=r'''
def check(fn):
    events = [
        {"account": "a", "delta": 10},
        {"account": "a", "delta": -10, "void": True},
        {"account": "b", "delta": 5},
        {"account": "a", "delta": -3},
        {"account": "b", "delta": -5},
    ]
    assert fn(events) == {"a": 7}
    assert fn([{"account": "x", "delta": 1}, {"account": "x", "delta": -1}]) == {}
''',
    ),
)


TOOL_TASKS: tuple[ToolExpertTask, ...] = (
    ToolExpertTask(
        task_id="cancel-reroute-only-active-gift",
        domain="retail",
        mechanism="multi_action_with_do_not_touch_distractor",
        instruction=(
            "Cancel duplicate order O-EX-1-DUP and change active gift order O-EX-1-GIFT to "
            "'81 Cypress Court'. Do not alter shipped order O-EX-1-OLD."
        ),
        initial_state={
            "orders": {
                "O-EX-1-DUP": {"customer_id": "C-EX-1", "status": "processing", "address": "10 Oak"},
                "O-EX-1-GIFT": {"customer_id": "C-EX-1", "status": "processing", "address": "10 Oak"},
                "O-EX-1-OLD": {"customer_id": "C-EX-1", "status": "shipped", "address": "10 Oak"},
            }
        },
        success=[
            {"path": ["orders", "O-EX-1-DUP", "status"], "equals": "cancelled"},
            {"path": ["orders", "O-EX-1-GIFT", "address"], "equals": "81 Cypress Court"},
            {"path": ["orders", "O-EX-1-OLD", "status"], "equals": "shipped"},
        ],
        allowed_tools=["cancel_order", "update_shipping_address", "finish"],
    ),
    ToolExpertTask(
        task_id="seat-two-active-ignore-standby",
        domain="airline",
        mechanism="parallel_updates_with_inactive_distractor",
        instruction=(
            "Assign active reservations R-EX-2-A and R-EX-2-B to seats 9A and 9B respectively. "
            "R-EX-2-S is standby/cancelled and must remain unchanged."
        ),
        initial_state={
            "reservations": {
                "R-EX-2-A": {"passenger_id": "P-EX-2-A", "status": "active", "seat": "12C"},
                "R-EX-2-B": {"passenger_id": "P-EX-2-B", "status": "active", "seat": None},
                "R-EX-2-S": {"passenger_id": "P-EX-2-S", "status": "cancelled", "seat": None},
            }
        },
        success=[
            {"path": ["reservations", "R-EX-2-A", "seat"], "equals": "9A"},
            {"path": ["reservations", "R-EX-2-B", "seat"], "equals": "9B"},
            {"path": ["reservations", "R-EX-2-S", "seat"], "equals": None},
        ],
        allowed_tools=["assign_seat", "finish"],
    ),
    ToolExpertTask(
        task_id="freeze-two-keep-virtual",
        domain="banking",
        mechanism="selective_multi_update",
        instruction=(
            "Freeze CARD-EX-3-DEBIT and CARD-EX-3-TRAVEL after wallet theft. Keep CARD-EX-3-VIRTUAL active."
        ),
        initial_state={
            "cards": {
                "CARD-EX-3-DEBIT": {"user_id": "U-EX-3", "status": "active"},
                "CARD-EX-3-TRAVEL": {"user_id": "U-EX-3", "status": "active"},
                "CARD-EX-3-VIRTUAL": {"user_id": "U-EX-3", "status": "active"},
            }
        },
        success=[
            {"path": ["cards", "CARD-EX-3-DEBIT", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-EX-3-TRAVEL", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-EX-3-VIRTUAL", "status"], "equals": "active"},
        ],
        allowed_tools=["freeze_card", "finish"],
    ),
    ToolExpertTask(
        task_id="reroute-two-unshipped-only",
        domain="retail",
        mechanism="bulk_update_with_shipped_guardrail",
        instruction=(
            "Update unshipped orders O-EX-4-A and O-EX-4-B to '500 Harbor Plaza'. "
            "Do not change shipped order O-EX-4-C."
        ),
        initial_state={
            "orders": {
                "O-EX-4-A": {"customer_id": "C-EX-4", "status": "processing", "address": "1 Pine"},
                "O-EX-4-B": {"customer_id": "C-EX-4", "status": "packed", "address": "1 Pine"},
                "O-EX-4-C": {"customer_id": "C-EX-4", "status": "shipped", "address": "1 Pine"},
            }
        },
        success=[
            {"path": ["orders", "O-EX-4-A", "address"], "equals": "500 Harbor Plaza"},
            {"path": ["orders", "O-EX-4-B", "address"], "equals": "500 Harbor Plaza"},
            {"path": ["orders", "O-EX-4-C", "address"], "equals": "1 Pine"},
        ],
        allowed_tools=["update_shipping_address", "finish"],
    ),
)


LONG_TASKS: tuple[LongExpertTask, ...] = (
    LongExpertTask(
        task_id="signed-revision-release-packet",
        mechanism="revision_precedence_with_canceled_record",
        prompt=(
            "Using only active signed records, return the Beacon packet exactly as "
            "`window / rollback owner / gate`. Return only that packet."
        ),
        must_contain=["2026-10-04 03:30 UTC / Omar Velez / gate-bc-42"],
        must_not_contain=["2026-09-27", "Mason Reed", "gate-bc-17", "canceled"],
        documents=[
            {"title": "rules", "text": "Signed records beat unsigned notes. Later signed revision wins. Ignore draft and canceled records."},
            {"title": "schedule r1", "text": "signed revision 1: Beacon window 2026-09-27 02:00 UTC."},
            {"title": "schedule r3", "text": "signed revision 3: Beacon window 2026-10-04 03:30 UTC."},
            {"title": "owner r1", "text": "signed revision 1: rollback owner Mason Reed."},
            {"title": "owner r4", "text": "signed revision 4: rollback owner Omar Velez."},
            {"title": "gate r1", "text": "signed revision 1: gate-bc-17."},
            {"title": "gate r5", "text": "signed revision 5: gate-bc-42."},
            {"title": "canceled", "text": "canceled signed record proposed 2026-10-05 and gate-bc-99."},
        ],
    ),
    LongExpertTask(
        task_id="cutoff-cargo-state",
        mechanism="temporal_cutoff_and_projection_rejection",
        prompt=(
            "Apply the cutoff at 2026-06-18T12:00Z. Return container B7 exactly as "
            "`location / hold reason / release code`. Return only that packet."
        ),
        must_contain=["Lyon / customs mismatch / RL-42"],
        must_not_contain=["Milan", "RL-18", "temperature audit", "B9"],
        documents=[
            {"title": "rules", "text": "Use active events at or before cutoff. Later events are projections. Holds remain until a release before cutoff."},
            {"title": "B7 01", "text": "2026-06-17T09:00Z active B7 arrived Milan."},
            {"title": "B7 02", "text": "2026-06-18T06:00Z active B7 arrived Lyon."},
            {"title": "B7 03", "text": "2026-06-18T08:15Z active B7 hold reason customs mismatch."},
            {"title": "B7 projection", "text": "2026-06-18T13:40Z projected B7 release code RL-18."},
            {"title": "release ledger", "text": "For customs mismatch at Lyon, active release code RL-42."},
            {"title": "B9", "text": "B9 is held in Lyon for temperature audit and uses RL-18."},
        ],
    ),
)


DIRECT_TASKS_V2: tuple[DirectExpertTask, ...] = (
    DirectExpertTask(
        task_id="appeal-eligibility-exception",
        domain="general",
        mechanism="unless_exception_with_audit_override",
        grader_type="mc_letter",
        answer="C",
        prompt=(
            "A grant appeal is eligible if the applicant submitted a signed denial letter and is not past "
            "the 30-day window, unless the file has an audit hold. A later director waiver can override "
            "only the 30-day window, not an audit hold. Case file: signed denial letter present; day 42; "
            "director waiver signed; audit hold active. What is the correct status?\n\n"
            "A. Eligible because the waiver overrides every blocker\n"
            "B. Eligible because the denial letter is signed\n"
            "C. Ineligible because audit hold is not waivable\n"
            "D. Ineligible only because the letter is unsigned\n\n"
            "Return only the letter."
        ),
    ),
    DirectExpertTask(
        task_id="weighted-sample-after-removal",
        domain="math",
        mechanism="conditional_probability_after_selective_removal",
        grader_type="math_equal",
        answer="5/13",
        prompt=(
            "A review queue has 9 red urgent, 4 blue urgent, 6 red normal, and 7 blue normal tickets. "
            "A cleanup removes all normal red tickets and exactly two urgent red tickets. From the remaining "
            "queue, a ticket is sampled uniformly. What is P(ticket is urgent and blue | ticket remains)? "
            "Return a reduced fraction."
        ),
    ),
    DirectExpertTask(
        task_id="measurement-calibration-causal",
        domain="science",
        mechanism="measurement_intervention_vs_state_change",
        grader_type="mc_letter",
        answer="B",
        prompt=(
            "A lab sensor reads high when either reagent X is present or the sensor offset is miscalibrated. "
            "A technician directly changes only the sensor offset and the reading drops to normal. Which "
            "conclusion is justified?\n\n"
            "A. Reagent X was definitely absent before calibration.\n"
            "B. The pre-calibration high reading cannot by itself identify reagent X as present.\n"
            "C. Reagent X must have been removed by changing the offset.\n"
            "D. The calibration proves reagent X was present.\n\n"
            "Return only the letter."
        ),
    ),
    DirectExpertTask(
        task_id="majority-with-recusal",
        domain="general",
        mechanism="counting_denominator_shift",
        grader_type="mc_letter",
        answer="D",
        prompt=(
            "A committee motion passes only if yes votes exceed half of non-recused members. There are "
            "11 members. Two recuse before voting. Votes are 5 yes, 4 no, and the two recused do not count. "
            "What is the result?\n\n"
            "A. Passes because 5 yes is a majority of 11\n"
            "B. Fails because 5 is not greater than half of 11\n"
            "C. Ties because recused members count as no\n"
            "D. Passes because 5 is greater than half of 9\n\n"
            "Return only the letter."
        ),
    ),
    DirectExpertTask(
        task_id="rate-limit-after-burst-credit",
        domain="math",
        mechanism="stateful_quota_arithmetic",
        grader_type="math_equal",
        answer="17",
        prompt=(
            "An API account starts with 40 calls. A migration credit adds 15 calls, but only after the first "
            "batch. Batch A uses 28 calls. Batch B tries 21 calls after the credit is applied. How many calls "
            "remain after accepting as much of batch B as possible? Return only the integer."
        ),
    ),
    DirectExpertTask(
        task_id="signed-change-order-scope",
        domain="general",
        mechanism="scope_limited_override",
        grader_type="mc_letter",
        answer="A",
        prompt=(
            "Contract precedence is signed change order > signed base schedule > vendor email. A signed "
            "change order moves only the database migration to Friday. The signed base schedule keeps the "
            "frontend release on Wednesday. A vendor email says both are Friday. What controls the frontend "
            "release?\n\n"
            "A. Wednesday, from the signed base schedule\n"
            "B. Friday, because any change order moves the whole project\n"
            "C. Friday, because the vendor email is newest\n"
            "D. No date, because the records conflict\n\n"
            "Return only the letter."
        ),
    ),
)


CODE_TASKS_V2: tuple[CodeExpertTask, ...] = (
    CodeExpertTask(
        task_id="invoice-aging-with-credits",
        mechanism="ordered_allocation_with_void_and_credit_traps",
        entry_point="open_invoice_balances",
        prompt=(
            "Write open_invoice_balances(invoices, payments). Invoices are dicts with id, due, amount. "
            "Payments are dicts with amount and optional void. Apply non-void payment amounts to open "
            "invoices from earliest due date to latest, preserving invoice id. Negative payment amounts are "
            "credits and increase the oldest open balance. Return {invoice_id: remaining_amount} for nonzero "
            "balances only."
        ),
        test=r'''
def check(fn):
    invoices = [
        {"id": "b", "due": "2026-02-01", "amount": 50},
        {"id": "a", "due": "2026-01-01", "amount": 30},
        {"id": "c", "due": "2026-03-01", "amount": 20},
    ]
    payments = [{"amount": 25}, {"amount": 10, "void": True}, {"amount": 40}]
    assert fn(invoices, payments) == {"c": 5}
    payments = [{"amount": 30}, {"amount": -7}, {"amount": 10}]
    assert fn(invoices, payments) == {"b": 47, "c": 20}
''',
    ),
    CodeExpertTask(
        task_id="feature-flags-last-write-delete",
        mechanism="last_write_wins_with_delete_and_order",
        entry_point="active_flags",
        prompt=(
            "Write active_flags(events). Each event has key, value, and optional delete. Last event for a key "
            "wins; delete removes the key. Return a list of (key, value) for active keys ordered by first time "
            "the key ever appeared, not by last update."
        ),
        test=r'''
def check(fn):
    events = [
        {"key": "a", "value": True},
        {"key": "b", "value": 1},
        {"key": "a", "value": False},
        {"key": "c", "value": "x"},
        {"key": "b", "delete": True},
        {"key": "b", "value": 3},
    ]
    assert fn(events) == [("a", False), ("b", 3), ("c", "x")]
    assert fn([{"key": "x", "value": 1}, {"key": "x", "delete": True}]) == []
''',
    ),
    CodeExpertTask(
        task_id="route-collapse-exact-backtracks",
        mechanism="stack_cancellation_not_set_dedup",
        entry_point="collapse_route",
        prompt=(
            "Write collapse_route(steps). Steps are cardinal moves N,S,E,W. Cancel only adjacent exact "
            "backtracks, repeatedly, using stack behavior. Do not reorder or globally cancel non-adjacent "
            "moves. Return the remaining step string."
        ),
        test=r'''
def check(fn):
    assert fn("NS") == ""
    assert fn("NESW") == "NESW"
    assert fn("NSEW") == ""
    assert fn("NNSS") == ""
    assert fn("NENWS") == "NENWS"
    assert fn("EWNSWE") == ""
''',
    ),
    CodeExpertTask(
        task_id="windowed-error-budget",
        mechanism="inclusive_exclusive_window_edges",
        entry_point="burned_minutes",
        prompt=(
            "Write burned_minutes(incidents, start, end). Incidents are (start_min, end_min, severity). "
            "Use half-open windows [start, end) and incident intervals [a, b). Count only severity 'page'. "
            "Return total overlapped minutes, counting overlapping page incidents only once."
        ),
        test=r'''
def check(fn):
    incidents = [(0, 10, "page"), (5, 15, "page"), (15, 20, "page"), (3, 8, "ticket")]
    assert fn(incidents, 0, 15) == 15
    assert fn(incidents, 10, 20) == 10
    assert fn([(1, 3, "page"), (3, 5, "page")], 1, 5) == 4
    assert fn([(0, 100, "ticket")], 0, 100) == 0
''',
    ),
    CodeExpertTask(
        task_id="csv-pipe-escaped-fields",
        mechanism="small_parser_with_escape_state",
        entry_point="parse_pipe_row",
        prompt=(
            "Write parse_pipe_row(text). Split a single row on unescaped | characters. Backslash escapes "
            "only backslash and pipe. Keep other backslashes literally. Raise ValueError on a dangling final "
            "backslash."
        ),
        test=r'''
def check(fn):
    assert fn(r"a|b|c") == ["a", "b", "c"]
    assert fn(r"a\|b|c") == ["a|b", "c"]
    assert fn(r"a\\|b") == [r"a\\", "b"]
    assert fn(r"a\q|b") == [r"a\q", "b"]
    try:
        fn("abc\\")
    except ValueError:
        pass
    else:
        raise AssertionError("dangling backslash must fail")
''',
    ),
    CodeExpertTask(
        task_id="inventory-snapshot-before-cutoff",
        mechanism="temporal_cutoff_and_cancellation",
        entry_point="inventory_at",
        prompt=(
            "Write inventory_at(events, cutoff). Events are dicts with sku, qty, ts, and optional cancel_id. "
            "Apply events with ts <= cutoff in input order. If an event has cancel_id, it cancels the earlier "
            "event with that id if that earlier event is already applied. Normal events may have id. Return "
            "nonzero sku quantities."
        ),
        test=r'''
def check(fn):
    events = [
        {"id": "e1", "sku": "A", "qty": 5, "ts": 1},
        {"id": "e2", "sku": "A", "qty": -2, "ts": 2},
        {"cancel_id": "e2", "sku": "A", "qty": 0, "ts": 3},
        {"id": "e3", "sku": "B", "qty": 7, "ts": 5},
        {"id": "e4", "sku": "A", "qty": 1, "ts": 6},
    ]
    assert fn(events, 4) == {"A": 5}
    assert fn(events, 6) == {"A": 6, "B": 7}
''',
    ),
    CodeExpertTask(
        task_id="title-case-small-words",
        mechanism="formatting_rules_with_position_exceptions",
        entry_point="headline_case",
        prompt=(
            "Write headline_case(title). Capitalize first and last words and words not in "
            "{a, an, the, and, or, but, of, in, on, for}. Preserve hyphenated compounds by applying the "
            "same rule to each hyphen part. Input is lowercase words separated by single spaces."
        ),
        test=r'''
def check(fn):
    assert fn("war and peace") == "War and Peace"
    assert fn("the state of the art") == "The State of the Art"
    assert fn("cost-benefit analysis of a plan") == "Cost-Benefit Analysis of a Plan"
    assert fn("in and out") == "In and Out"
''',
    ),
    CodeExpertTask(
        task_id="tiered-shipping-after-discount",
        mechanism="order_of_operations_boundary",
        entry_point="shipping_fee",
        prompt=(
            "Write shipping_fee(subtotal_cents, coupon_cents). Apply coupon first, floor at zero, then "
            "shipping is 0 if discounted subtotal >= 5000, 499 if discounted subtotal >= 2000, otherwise 899."
        ),
        test=r'''
def check(fn):
    assert fn(6000, 1500) == 499
    assert fn(5000, 0) == 0
    assert fn(2500, 600) == 899
    assert fn(1000, 5000) == 899
''',
    ),
)


TOOL_TASKS_V2: tuple[ToolExpertTask, ...] = (
    ToolExpertTask(
        task_id="retail-split-cancel-reroute-four",
        domain="retail",
        mechanism="longer_action_chain_with_shipped_and_duplicate_distractors",
        instruction=(
            "Cancel processing duplicate orders O-V2-1-D1 and O-V2-1-D2. Update only active gift "
            "orders O-V2-1-G1 and O-V2-1-G2 to '44 Juniper Lane'. Leave shipped order O-V2-1-S unchanged."
        ),
        initial_state={
            "orders": {
                "O-V2-1-D1": {"customer_id": "C-V2-1", "status": "processing", "address": "9 Old"},
                "O-V2-1-D2": {"customer_id": "C-V2-1", "status": "packed", "address": "9 Old"},
                "O-V2-1-G1": {"customer_id": "C-V2-1", "status": "processing", "address": "9 Old"},
                "O-V2-1-G2": {"customer_id": "C-V2-1", "status": "processing", "address": "9 Old"},
                "O-V2-1-S": {"customer_id": "C-V2-1", "status": "shipped", "address": "9 Old"},
            }
        },
        success=[
            {"path": ["orders", "O-V2-1-D1", "status"], "equals": "cancelled"},
            {"path": ["orders", "O-V2-1-D2", "status"], "equals": "cancelled"},
            {"path": ["orders", "O-V2-1-G1", "address"], "equals": "44 Juniper Lane"},
            {"path": ["orders", "O-V2-1-G2", "address"], "equals": "44 Juniper Lane"},
            {"path": ["orders", "O-V2-1-S", "address"], "equals": "9 Old"},
        ],
        allowed_tools=["cancel_order", "update_shipping_address", "finish"],
        max_turns=10,
    ),
    ToolExpertTask(
        task_id="airline-family-seat-swap-ignore-cancelled",
        domain="airline",
        mechanism="multi_entity_assignment_with_inactive_guardrail",
        instruction=(
            "Seat active reservations R-V2-2-A, R-V2-2-B, and R-V2-2-C together at 14A, 14B, and 14C. "
            "Do not assign seats to cancelled reservation R-V2-2-X."
        ),
        initial_state={
            "reservations": {
                "R-V2-2-A": {"passenger_id": "P-A", "status": "active", "seat": None},
                "R-V2-2-B": {"passenger_id": "P-B", "status": "active", "seat": "22D"},
                "R-V2-2-C": {"passenger_id": "P-C", "status": "active", "seat": None},
                "R-V2-2-X": {"passenger_id": "P-X", "status": "cancelled", "seat": None},
            }
        },
        success=[
            {"path": ["reservations", "R-V2-2-A", "seat"], "equals": "14A"},
            {"path": ["reservations", "R-V2-2-B", "seat"], "equals": "14B"},
            {"path": ["reservations", "R-V2-2-C", "seat"], "equals": "14C"},
            {"path": ["reservations", "R-V2-2-X", "seat"], "equals": None},
        ],
        allowed_tools=["assign_seat", "finish"],
        max_turns=8,
    ),
    ToolExpertTask(
        task_id="bank-freeze-physical-leave-virtual-and-closed",
        domain="banking",
        mechanism="selective_bulk_update_with_status_distractors",
        instruction=(
            "After mailbox theft, freeze physical active cards CARD-V2-3-D and CARD-V2-3-T. "
            "Leave virtual CARD-V2-3-V active and closed CARD-V2-3-C closed."
        ),
        initial_state={
            "cards": {
                "CARD-V2-3-D": {"user_id": "U-V2-3", "status": "active", "kind": "debit"},
                "CARD-V2-3-T": {"user_id": "U-V2-3", "status": "active", "kind": "travel"},
                "CARD-V2-3-V": {"user_id": "U-V2-3", "status": "active", "kind": "virtual"},
                "CARD-V2-3-C": {"user_id": "U-V2-3", "status": "closed", "kind": "debit"},
            }
        },
        success=[
            {"path": ["cards", "CARD-V2-3-D", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-V2-3-T", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-V2-3-V", "status"], "equals": "active"},
            {"path": ["cards", "CARD-V2-3-C", "status"], "equals": "closed"},
        ],
        allowed_tools=["freeze_card", "finish"],
        max_turns=8,
    ),
    ToolExpertTask(
        task_id="retail-mixed-status-three-active",
        domain="retail",
        mechanism="bulk_address_update_with_partial_status_filter",
        instruction=(
            "Update only unshipped active orders O-V2-4-A, O-V2-4-B, and O-V2-4-C to "
            "'12 Market Arcade'. Cancel duplicate O-V2-4-D. Do not change shipped O-V2-4-S."
        ),
        initial_state={
            "orders": {
                "O-V2-4-A": {"customer_id": "C-V2-4", "status": "processing", "address": "1 Main"},
                "O-V2-4-B": {"customer_id": "C-V2-4", "status": "packed", "address": "1 Main"},
                "O-V2-4-C": {"customer_id": "C-V2-4", "status": "processing", "address": "1 Main"},
                "O-V2-4-D": {"customer_id": "C-V2-4", "status": "processing", "address": "1 Main"},
                "O-V2-4-S": {"customer_id": "C-V2-4", "status": "shipped", "address": "1 Main"},
            }
        },
        success=[
            {"path": ["orders", "O-V2-4-A", "address"], "equals": "12 Market Arcade"},
            {"path": ["orders", "O-V2-4-B", "address"], "equals": "12 Market Arcade"},
            {"path": ["orders", "O-V2-4-C", "address"], "equals": "12 Market Arcade"},
            {"path": ["orders", "O-V2-4-D", "status"], "equals": "cancelled"},
            {"path": ["orders", "O-V2-4-S", "address"], "equals": "1 Main"},
        ],
        allowed_tools=["update_shipping_address", "cancel_order", "finish"],
        max_turns=10,
    ),
    ToolExpertTask(
        task_id="airline-two-couples-crossed-assignments",
        domain="airline",
        mechanism="paired_assignment_with_order_trap",
        instruction=(
            "Assign R-V2-5-A and R-V2-5-B to 18A and 18B. Assign R-V2-5-C and R-V2-5-D to 19A and 19B. "
            "Do not touch cancelled R-V2-5-Z."
        ),
        initial_state={
            "reservations": {
                "R-V2-5-A": {"passenger_id": "P-A", "status": "active", "seat": None},
                "R-V2-5-B": {"passenger_id": "P-B", "status": "active", "seat": None},
                "R-V2-5-C": {"passenger_id": "P-C", "status": "active", "seat": None},
                "R-V2-5-D": {"passenger_id": "P-D", "status": "active", "seat": None},
                "R-V2-5-Z": {"passenger_id": "P-Z", "status": "cancelled", "seat": None},
            }
        },
        success=[
            {"path": ["reservations", "R-V2-5-A", "seat"], "equals": "18A"},
            {"path": ["reservations", "R-V2-5-B", "seat"], "equals": "18B"},
            {"path": ["reservations", "R-V2-5-C", "seat"], "equals": "19A"},
            {"path": ["reservations", "R-V2-5-D", "seat"], "equals": "19B"},
            {"path": ["reservations", "R-V2-5-Z", "seat"], "equals": None},
        ],
        allowed_tools=["assign_seat", "finish"],
        max_turns=10,
    ),
    ToolExpertTask(
        task_id="bank-freeze-all-compromised-except-new",
        domain="banking",
        mechanism="exception_after_bulk_selection",
        instruction=(
            "Freeze compromised cards CARD-V2-6-A, CARD-V2-6-B, and CARD-V2-6-C. "
            "Do not freeze replacement card CARD-V2-6-NEW."
        ),
        initial_state={
            "cards": {
                "CARD-V2-6-A": {"user_id": "U-V2-6", "status": "active"},
                "CARD-V2-6-B": {"user_id": "U-V2-6", "status": "active"},
                "CARD-V2-6-C": {"user_id": "U-V2-6", "status": "active"},
                "CARD-V2-6-NEW": {"user_id": "U-V2-6", "status": "active"},
            }
        },
        success=[
            {"path": ["cards", "CARD-V2-6-A", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-V2-6-B", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-V2-6-C", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-V2-6-NEW", "status"], "equals": "active"},
        ],
        allowed_tools=["freeze_card", "finish"],
        max_turns=8,
    ),
)


LONG_TASKS_V2: tuple[LongExpertTask, ...] = (
    LongExpertTask(
        task_id="incident-bridge-conflicting-amendments",
        mechanism="multi_document_precedence_and_component_scope",
        prompt=(
            "Using only active signed records, return the final Atlas incident packet exactly as "
            "`primary owner / database window / frontend window / rollback code`. Return only that packet."
        ),
        must_contain=["Iris Chen / 2026-11-08 01:00 UTC / 2026-11-06 22:00 UTC / RB-73"],
        must_not_contain=["Noah Patel", "2026-11-09", "2026-11-07 20:00", "RB-19", "draft"],
        documents=[
            {"title": "rules", "text": "Signed active amendments override signed base records only for their named component. Draft and canceled records do not apply."},
            {"title": "base owner", "text": "signed active base: primary owner Noah Patel."},
            {"title": "owner amendment", "text": "signed active amendment: primary owner Iris Chen for all Atlas incident work."},
            {"title": "base db", "text": "signed active base: database window 2026-11-07 20:00 UTC."},
            {"title": "db amendment", "text": "signed active amendment: database window 2026-11-08 01:00 UTC."},
            {"title": "frontend base", "text": "signed active base: frontend window 2026-11-06 22:00 UTC."},
            {"title": "frontend draft", "text": "draft amendment: frontend window 2026-11-09 03:00 UTC."},
            {"title": "rollback", "text": "signed active amendment: rollback code RB-73."},
            {"title": "canceled rollback", "text": "canceled signed record: rollback code RB-19."},
        ],
    ),
    LongExpertTask(
        task_id="clinical-note-cutoff-medication",
        mechanism="temporal_cutoff_and_discontinued_medication",
        prompt=(
            "At cutoff 2026-04-12T10:00Z, return patient K21 exactly as "
            "`active medication / held medication / follow-up owner`. Return only that packet."
        ),
        must_contain=["ambrisentan / warfarin / Dr. Vale"],
        must_not_contain=["sildenafil", "Dr. Noor", "projected", "restart warfarin"],
        documents=[
            {"title": "rules", "text": "Use signed entries at or before cutoff. Later entries are projections. Discontinued medications are not active. Holds remain active until a signed release before cutoff."},
            {"title": "entry 1", "text": "2026-04-10T08:00Z signed: K21 active medication sildenafil."},
            {"title": "entry 2", "text": "2026-04-11T09:00Z signed: discontinue sildenafil; start ambrisentan."},
            {"title": "entry 3", "text": "2026-04-11T12:30Z signed: hold warfarin."},
            {"title": "entry 4", "text": "2026-04-12T09:30Z signed: follow-up owner Dr. Vale."},
            {"title": "entry 5", "text": "2026-04-12T11:00Z projected: restart warfarin and assign Dr. Noor."},
        ],
    ),
    LongExpertTask(
        task_id="procurement-award-scope",
        mechanism="vendor_scope_limited_override",
        prompt=(
            "Return Lot-7 award exactly as `hardware vendor / support vendor / escalation tier`. "
            "Use active signed records only. Return only that packet."
        ),
        must_contain=["HelioWorks / Northstar Support / tier-2"],
        must_not_contain=["Aster Labs", "tier-3", "OmniDesk", "unsigned"],
        documents=[
            {"title": "rules", "text": "Signed amendments override base awards only for the named lot and category. Unsigned procurement notes do not apply."},
            {"title": "base lot7", "text": "signed base award Lot-7: hardware vendor Aster Labs; support vendor Northstar Support; escalation tier tier-2."},
            {"title": "hardware amendment", "text": "signed amendment Lot-7 hardware only: hardware vendor HelioWorks."},
            {"title": "support note", "text": "unsigned note Lot-7 support: support vendor OmniDesk; escalation tier tier-3."},
            {"title": "lot8", "text": "signed amendment Lot-8: hardware vendor OmniDesk; escalation tier tier-3."},
        ],
    ),
    LongExpertTask(
        task_id="memory-handoff-latest-accepted",
        mechanism="accepted_rejected_memory_handoff",
        prompt=(
            "For project Lumen, return exactly `accepted hypothesis / rejected hypothesis / next experiment`. "
            "Use only accepted lab notebook entries. Return only that packet."
        ),
        must_contain=["cache eviction threshold drift / GPU clock throttling / run cold-start sweep"],
        must_not_contain=["network jitter", "increase batch size", "thermal paste", "rejected"],
        documents=[
            {"title": "rules", "text": "Accepted entries are usable. Rejected entries are historical distractors even when newer."},
            {"title": "entry a", "text": "accepted: hypothesis network jitter; next experiment packet capture."},
            {"title": "entry b", "text": "accepted: rejected hypothesis network jitter; accepted hypothesis cache eviction threshold drift."},
            {"title": "entry c", "text": "rejected: hypothesis GPU clock throttling; next experiment increase batch size."},
            {"title": "entry d", "text": "accepted: rejected hypothesis GPU clock throttling; next experiment run cold-start sweep."},
            {"title": "entry e", "text": "rejected: hypothesis thermal paste; next experiment replace heatsink."},
        ],
    ),
)


def _source_ref(*, source_name: str = SOURCE_NAME, source_version: str = SOURCE_VERSION) -> SourceRef:
    return SourceRef(
        name=source_name,
        version=source_version,
        policy="train_allowed",
        url_or_ref="expert-designed://fugu-ultra-disagreement",
    )


def _base_metadata(domain: str, subdomain: str, mechanism: str) -> TaskMetadata:
    return TaskMetadata(
        domain=domain,
        subdomain=subdomain,
        difficulty_estimate=0.8,
        tags=["expert-designed", "disagreement-targeted", mechanism],
        requires_tools=subdomain == "tool_dialogue",
        requires_long_context=subdomain == "long_context",
    )


def _tools_for(allowed_tools: list[str]) -> list[dict[str, Any]]:
    allowed = set(allowed_tools)
    return [tool for tool in COMMON_TOOLS if tool["function"]["name"] in allowed]


def _source_identity(version: str) -> tuple[str, str]:
    if version == "v1":
        return SOURCE_NAME, SOURCE_VERSION
    if version == "v2":
        return SOURCE_NAME_V2, SOURCE_VERSION_V2
    raise ValueError(f"unknown expert disagreement version {version!r}")


def _task_sets(version: str) -> tuple[
    tuple[DirectExpertTask, ...],
    tuple[CodeExpertTask, ...],
    tuple[ToolExpertTask, ...],
    tuple[LongExpertTask, ...],
]:
    if version == "v1":
        return DIRECT_TASKS, CODE_TASKS, TOOL_TASKS, LONG_TASKS
    if version == "v2":
        return DIRECT_TASKS_V2, CODE_TASKS_V2, TOOL_TASKS_V2, LONG_TASKS_V2
    raise ValueError(f"unknown expert disagreement version {version!r}")


def direct_task_spec(
    task: DirectExpertTask,
    *,
    source_name: str = SOURCE_NAME,
    source_version: str = SOURCE_VERSION,
) -> TaskSpec:
    task_id = f"{source_name}__direct__{task.task_id}"
    group = f"{source_name}/direct/{task.task_id}"
    return TaskSpec(
        task_id=task_id,
        capability=task.domain,
        source=_source_ref(source_name=source_name, source_version=source_version),
        input=TaskInput(messages=[{"role": "user", "content": task.prompt}]),
        environment=EnvironmentSpec(harness="direct_qa", wall_time_seconds=180),
        grader=GraderSpec(type=task.grader_type, expected_answer=task.answer),
        splitting=SplittingSpec(group_id=group, split="grpo_train", contamination_group=group),
        metadata=_base_metadata(task.domain, "direct_reasoning", task.mechanism),
    )


def code_task_spec(
    task: CodeExpertTask,
    *,
    source_name: str = SOURCE_NAME,
    source_version: str = SOURCE_VERSION,
) -> TaskSpec:
    task_id = f"{source_name}__code__{task.task_id}"
    group = f"{source_name}/code/{task.task_id}"
    return TaskSpec(
        task_id=task_id,
        capability="code",
        source=_source_ref(source_name=source_name, source_version=source_version),
        input=TaskInput(messages=[{"role": "user", "content": task.prompt}]),
        environment=EnvironmentSpec(harness="code_exec", wall_time_seconds=180),
        grader=GraderSpec(
            type="code_exec",
            expected_answer={"test": task.test, "entry_point": task.entry_point, "timeout": 5},
        ),
        splitting=SplittingSpec(group_id=group, split="grpo_train", contamination_group=group),
        metadata=_base_metadata("code", "unit_code", task.mechanism),
    )


def tool_task_spec(
    task: ToolExpertTask,
    *,
    source_name: str = SOURCE_NAME,
    source_version: str = SOURCE_VERSION,
) -> TaskSpec:
    task_id = f"{source_name}__tool__{task.task_id}"
    group = f"{source_name}/tool/{task.task_id}"
    return TaskSpec(
        task_id=task_id,
        capability="tool_dialogue",
        source=_source_ref(source_name=source_name, source_version=source_version),
        input=TaskInput(messages=[{"role": "user", "content": task.instruction}], tools=_tools_for(task.allowed_tools)),
        environment=EnvironmentSpec(harness="tool_dialog", wall_time_seconds=300),
        grader=GraderSpec(
            type="db_state",
            expected_answer={
                "domain": task.domain,
                "initial_state": task.initial_state,
                "success": task.success,
                "allowed_tools": task.allowed_tools,
                "max_turns": task.max_turns,
            },
        ),
        splitting=SplittingSpec(group_id=group, split="grpo_train", contamination_group=group),
        metadata=_base_metadata(task.domain, "tool_dialogue", task.mechanism),
    )


def long_task_spec(
    task: LongExpertTask,
    *,
    source_name: str = SOURCE_NAME,
    source_version: str = SOURCE_VERSION,
) -> TaskSpec:
    task_id = f"{source_name}__long__{task.task_id}"
    group = f"{source_name}/long/{task.task_id}"
    return TaskSpec(
        task_id=task_id,
        capability="long_context",
        source=_source_ref(source_name=source_name, source_version=source_version),
        input=TaskInput(messages=[{"role": "user", "content": task.prompt}], context_documents=task.documents),
        environment=EnvironmentSpec(harness="long_context", wall_time_seconds=300),
        grader=GraderSpec(
            type="contains_all_absent",
            expected_answer={"must_contain": task.must_contain, "must_not_contain": task.must_not_contain},
        ),
        splitting=SplittingSpec(group_id=group, split="grpo_train", contamination_group=group),
        metadata=_base_metadata("long_context", "long_context", task.mechanism),
    )


def task_specs(version: str = "v1") -> list[TaskSpec]:
    source_name, source_version = _source_identity(version)
    direct_tasks, code_tasks, tool_tasks, long_tasks = _task_sets(version)
    return [
        *(direct_task_spec(task, source_name=source_name, source_version=source_version) for task in direct_tasks),
        *(code_task_spec(task, source_name=source_name, source_version=source_version) for task in code_tasks),
        *(tool_task_spec(task, source_name=source_name, source_version=source_version) for task in tool_tasks),
        *(long_task_spec(task, source_name=source_name, source_version=source_version) for task in long_tasks),
    ]


def materialize_expert_disagreement_tasks(
    *,
    out_jsonl: Path,
    report_out: Path | None = None,
    limit: int | None = None,
    version: str = "v1",
) -> dict[str, Any]:
    source_name, _source_version = _source_identity(version)
    specs = task_specs(version=version)
    if limit is not None:
        specs = specs[:limit]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for spec in specs:
            f.write(json.dumps(spec.model_dump(mode="json"), sort_keys=True) + "\n")

    lane_counts: dict[str, int] = {}
    mechanisms: dict[str, list[str]] = {}
    for spec in specs:
        harness: Literal["direct_qa", "code_exec", "tool_dialog", "long_context"] = spec.environment.harness  # type: ignore[assignment]
        lane = {
            "direct_qa": "math_science_knowledge",
            "code_exec": "unit_and_scientific_code",
            "tool_dialog": "tool_dialogue",
            "long_context": "long_context_memory_planning",
        }[harness]
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        mechanisms[spec.task_id] = [tag for tag in spec.metadata.tags if tag not in {"expert-designed", "disagreement-targeted"}]

    report = {
        "version": f"expert_disagreement_tasks_{version}",
        "source": source_name,
        "purpose": "Expert-designed verifier-backed text tasks intended to induce worker/workflow disagreement.",
        "task_count": len(specs),
        "lane_counts": dict(sorted(lane_counts.items())),
        "mechanisms": mechanisms,
        "out_jsonl": str(out_jsonl),
        "splits": sorted({spec.splitting.split for spec in specs}),
        "live_calls": False,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
