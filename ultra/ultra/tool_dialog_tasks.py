"""Small deterministic tau-style tool-dialogue tasks.

These are generated custom domains, not official tau-bench eval seeds. They give the
Ultra data mix a runnable tool-dialogue lane with deterministic DB-state grading.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)

SOURCE_NAME = "tau_custom"
SOURCE_VERSION = "v1"


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


COMMON_TOOLS = [
    _tool(
        "cancel_order",
        "Cancel a retail order that has not shipped.",
        {"order_id": {"type": "string"}},
        ["order_id"],
    ),
    _tool(
        "update_shipping_address",
        "Update a shipping address for an order that has not shipped.",
        {"order_id": {"type": "string"}, "address": {"type": "string"}},
        ["order_id", "address"],
    ),
    _tool(
        "assign_seat",
        "Assign a passenger seat on an active reservation.",
        {"reservation_id": {"type": "string"}, "seat": {"type": "string"}},
        ["reservation_id", "seat"],
    ),
    _tool(
        "freeze_card",
        "Freeze an active payment card.",
        {"card_id": {"type": "string"}},
        ["card_id"],
    ),
    _tool(
        "finish",
        "Finish after the requested state change is complete.",
        {},
        [],
    ),
]


@dataclass(frozen=True)
class ToolDialogTask:
    domain: str
    task_id: str
    instruction: str
    initial_state: dict[str, Any]
    success: list[dict[str, Any]]
    allowed_tools: list[str]
    max_turns: int = 4


_SEED_TASKS: tuple[ToolDialogTask, ...] = (
    ToolDialogTask(
        domain="retail",
        task_id="cancel-pending-order",
        instruction=(
            "Customer C-100 asks to cancel order O-100. The order has not shipped. "
            "Use the tools to complete the request."
        ),
        initial_state={
            "orders": {
                "O-100": {"customer_id": "C-100", "status": "processing", "address": "9 Pine St"},
                "O-101": {"customer_id": "C-100", "status": "shipped", "address": "9 Pine St"},
            }
        },
        success=[{"path": ["orders", "O-100", "status"], "equals": "cancelled"}],
        allowed_tools=["cancel_order", "finish"],
    ),
    ToolDialogTask(
        domain="retail",
        task_id="update-unshipped-address",
        instruction=(
            "Customer C-200 moved. Update order O-200 shipping address to "
            "'42 Market St, Apt 5' before it ships."
        ),
        initial_state={
            "orders": {
                "O-200": {"customer_id": "C-200", "status": "processing", "address": "1 Old Rd"}
            }
        },
        success=[{"path": ["orders", "O-200", "address"], "equals": "42 Market St, Apt 5"}],
        allowed_tools=["update_shipping_address", "finish"],
    ),
    ToolDialogTask(
        domain="airline",
        task_id="assign-window-seat",
        instruction="Passenger P-300 wants seat 12A on reservation R-300. Assign that seat.",
        initial_state={
            "reservations": {
                "R-300": {"passenger_id": "P-300", "status": "active", "seat": "14C"}
            }
        },
        success=[{"path": ["reservations", "R-300", "seat"], "equals": "12A"}],
        allowed_tools=["assign_seat", "finish"],
    ),
    ToolDialogTask(
        domain="banking",
        task_id="freeze-lost-card",
        instruction="User U-400 reports card CARD-400 lost. Freeze the card.",
        initial_state={
            "cards": {
                "CARD-400": {"user_id": "U-400", "status": "active"},
                "CARD-401": {"user_id": "U-400", "status": "active"},
            }
        },
        success=[{"path": ["cards", "CARD-400", "status"], "equals": "frozen"}],
        allowed_tools=["freeze_card", "finish"],
    ),
    ToolDialogTask(
        domain="retail",
        task_id="cancel-duplicate-order",
        instruction="Customer C-210 accidentally placed duplicate order O-210. Cancel O-210 only.",
        initial_state={
            "orders": {
                "O-210": {"customer_id": "C-210", "status": "processing", "address": "18 Lake Ave"},
                "O-211": {"customer_id": "C-210", "status": "processing", "address": "18 Lake Ave"},
            }
        },
        success=[
            {"path": ["orders", "O-210", "status"], "equals": "cancelled"},
            {"path": ["orders", "O-211", "status"], "equals": "processing"},
        ],
        allowed_tools=["cancel_order", "finish"],
    ),
    ToolDialogTask(
        domain="retail",
        task_id="cancel-gift-order",
        instruction="Customer C-220 says gift order O-220 is no longer needed. Cancel it.",
        initial_state={
            "orders": {
                "O-220": {"customer_id": "C-220", "status": "packed", "address": "77 Birch Blvd"},
                "O-221": {"customer_id": "C-221", "status": "processing", "address": "4 Cedar Ct"},
            }
        },
        success=[{"path": ["orders", "O-220", "status"], "equals": "cancelled"}],
        allowed_tools=["cancel_order", "finish"],
    ),
    ToolDialogTask(
        domain="retail",
        task_id="update-office-address",
        instruction="Update order O-230 shipping address to '500 Office Park, Suite 12'.",
        initial_state={
            "orders": {
                "O-230": {"customer_id": "C-230", "status": "processing", "address": "12 Home St"},
                "O-231": {"customer_id": "C-230", "status": "shipped", "address": "12 Home St"},
            }
        },
        success=[{"path": ["orders", "O-230", "address"], "equals": "500 Office Park, Suite 12"}],
        allowed_tools=["update_shipping_address", "finish"],
    ),
    ToolDialogTask(
        domain="retail",
        task_id="update-rural-address",
        instruction="Customer C-240 needs order O-240 sent to '8 County Road 6'. Update the address.",
        initial_state={
            "orders": {
                "O-240": {"customer_id": "C-240", "status": "processing", "address": "8 County Road 5"}
            }
        },
        success=[{"path": ["orders", "O-240", "address"], "equals": "8 County Road 6"}],
        allowed_tools=["update_shipping_address", "finish"],
    ),
    ToolDialogTask(
        domain="airline",
        task_id="assign-aisle-seat",
        instruction="Passenger P-310 asks for aisle seat 18C on reservation R-310. Assign 18C.",
        initial_state={
            "reservations": {
                "R-310": {"passenger_id": "P-310", "status": "active", "seat": "20B"}
            }
        },
        success=[{"path": ["reservations", "R-310", "seat"], "equals": "18C"}],
        allowed_tools=["assign_seat", "finish"],
    ),
    ToolDialogTask(
        domain="airline",
        task_id="assign-exit-row-seat",
        instruction="Reservation R-320 is active. Assign passenger P-320 seat 21F.",
        initial_state={
            "reservations": {
                "R-320": {"passenger_id": "P-320", "status": "active", "seat": None},
                "R-321": {"passenger_id": "P-321", "status": "cancelled", "seat": None},
            }
        },
        success=[{"path": ["reservations", "R-320", "seat"], "equals": "21F"}],
        allowed_tools=["assign_seat", "finish"],
    ),
    ToolDialogTask(
        domain="airline",
        task_id="assign-connecting-flight-seat",
        instruction="Passenger P-330 wants seat 7D on active reservation R-330. Assign the seat.",
        initial_state={
            "reservations": {
                "R-330": {"passenger_id": "P-330", "status": "active", "seat": "7E"}
            }
        },
        success=[{"path": ["reservations", "R-330", "seat"], "equals": "7D"}],
        allowed_tools=["assign_seat", "finish"],
    ),
    ToolDialogTask(
        domain="banking",
        task_id="freeze-fraud-card",
        instruction="User U-410 reports fraudulent activity on CARD-410. Freeze CARD-410.",
        initial_state={
            "cards": {
                "CARD-410": {"user_id": "U-410", "status": "active"},
                "CARD-411": {"user_id": "U-410", "status": "active"},
            }
        },
        success=[
            {"path": ["cards", "CARD-410", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-411", "status"], "equals": "active"},
        ],
        allowed_tools=["freeze_card", "finish"],
    ),
    ToolDialogTask(
        domain="banking",
        task_id="freeze-travel-card",
        instruction="Freeze CARD-420 for user U-420 after a travel theft report.",
        initial_state={
            "cards": {
                "CARD-420": {"user_id": "U-420", "status": "active"}
            }
        },
        success=[{"path": ["cards", "CARD-420", "status"], "equals": "frozen"}],
        allowed_tools=["freeze_card", "finish"],
    ),
    ToolDialogTask(
        domain="banking",
        task_id="freeze-secondary-card",
        instruction="User U-430 asks to freeze secondary card CARD-431, not primary CARD-430.",
        initial_state={
            "cards": {
                "CARD-430": {"user_id": "U-430", "status": "active"},
                "CARD-431": {"user_id": "U-430", "status": "active"},
            }
        },
        success=[
            {"path": ["cards", "CARD-431", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-430", "status"], "equals": "active"},
        ],
        allowed_tools=["freeze_card", "finish"],
    ),
    ToolDialogTask(
        domain="retail",
        task_id="update-recipient-address",
        instruction="Order O-250 is a gift. Change its shipping address to '91 Maple Terrace'.",
        initial_state={
            "orders": {
                "O-250": {"customer_id": "C-250", "status": "processing", "address": "90 Maple Terrace"}
            }
        },
        success=[{"path": ["orders", "O-250", "address"], "equals": "91 Maple Terrace"}],
        allowed_tools=["update_shipping_address", "finish"],
    ),
    ToolDialogTask(
        domain="airline",
        task_id="assign-bulkhead-seat",
        instruction="Assign seat 2A to passenger P-340 on active reservation R-340.",
        initial_state={
            "reservations": {
                "R-340": {"passenger_id": "P-340", "status": "active", "seat": "3A"}
            }
        },
        success=[{"path": ["reservations", "R-340", "seat"], "equals": "2A"}],
        allowed_tools=["assign_seat", "finish"],
    ),
)


_HARD_TASKS: tuple[ToolDialogTask, ...] = (
    ToolDialogTask(
        domain="retail",
        task_id="hard-cancel-and-reroute-gift",
        instruction=(
            "Customer C-910 needs two changes. Cancel duplicate order O-910-DUP, "
            "then update active gift order O-910-GIFT to ship to '44 Juniper Lane'. "
            "Do not alter shipped order O-910-OLD."
        ),
        initial_state={
            "orders": {
                "O-910-DUP": {"customer_id": "C-910", "status": "processing", "address": "12 Cedar St"},
                "O-910-GIFT": {"customer_id": "C-910", "status": "processing", "address": "12 Cedar St"},
                "O-910-OLD": {"customer_id": "C-910", "status": "shipped", "address": "9 Old Rd"},
            }
        },
        success=[
            {"path": ["orders", "O-910-DUP", "status"], "equals": "cancelled"},
            {"path": ["orders", "O-910-GIFT", "address"], "equals": "44 Juniper Lane"},
            {"path": ["orders", "O-910-OLD", "status"], "equals": "shipped"},
        ],
        allowed_tools=["cancel_order", "update_shipping_address", "finish"],
        max_turns=7,
    ),
    ToolDialogTask(
        domain="retail",
        task_id="hard-reroute-two-open-orders",
        instruction=(
            "Customer C-920 moved offices. Update both unshipped orders O-920-A and O-920-B "
            "to '800 Harbor Plaza, Floor 6'. Leave already shipped order O-920-C untouched."
        ),
        initial_state={
            "orders": {
                "O-920-A": {"customer_id": "C-920", "status": "processing", "address": "1 North St"},
                "O-920-B": {"customer_id": "C-920", "status": "packed", "address": "1 North St"},
                "O-920-C": {"customer_id": "C-920", "status": "shipped", "address": "1 North St"},
            }
        },
        success=[
            {"path": ["orders", "O-920-A", "address"], "equals": "800 Harbor Plaza, Floor 6"},
            {"path": ["orders", "O-920-B", "address"], "equals": "800 Harbor Plaza, Floor 6"},
            {"path": ["orders", "O-920-C", "address"], "equals": "1 North St"},
        ],
        allowed_tools=["update_shipping_address", "finish"],
        max_turns=7,
    ),
    ToolDialogTask(
        domain="airline",
        task_id="hard-seat-two-passengers",
        instruction=(
            "Seat both travelers on the active reservations: assign R-930-A to 6A and R-930-B to 6B. "
            "Do not change cancelled reservation R-930-C."
        ),
        initial_state={
            "reservations": {
                "R-930-A": {"passenger_id": "P-930-A", "status": "active", "seat": "8C"},
                "R-930-B": {"passenger_id": "P-930-B", "status": "active", "seat": None},
                "R-930-C": {"passenger_id": "P-930-C", "status": "cancelled", "seat": None},
            }
        },
        success=[
            {"path": ["reservations", "R-930-A", "seat"], "equals": "6A"},
            {"path": ["reservations", "R-930-B", "seat"], "equals": "6B"},
            {"path": ["reservations", "R-930-C", "seat"], "equals": None},
        ],
        allowed_tools=["assign_seat", "finish"],
        max_turns=7,
    ),
    ToolDialogTask(
        domain="airline",
        task_id="hard-family-seat-split",
        instruction=(
            "Family reservation update: assign R-940-ADULT to 12C and R-940-CHILD to 12D. "
            "The standby reservation R-940-STANDBY is not active and must not be changed."
        ),
        initial_state={
            "reservations": {
                "R-940-ADULT": {"passenger_id": "P-940-A", "status": "active", "seat": "14C"},
                "R-940-CHILD": {"passenger_id": "P-940-C", "status": "active", "seat": "14D"},
                "R-940-STANDBY": {"passenger_id": "P-940-S", "status": "cancelled", "seat": None},
            }
        },
        success=[
            {"path": ["reservations", "R-940-ADULT", "seat"], "equals": "12C"},
            {"path": ["reservations", "R-940-CHILD", "seat"], "equals": "12D"},
            {"path": ["reservations", "R-940-STANDBY", "seat"], "equals": None},
        ],
        allowed_tools=["assign_seat", "finish"],
        max_turns=7,
    ),
    ToolDialogTask(
        domain="banking",
        task_id="hard-freeze-two-compromised-cards",
        instruction=(
            "User U-950 reports wallet theft. Freeze CARD-950-DEBIT and CARD-950-TRAVEL. "
            "Leave CARD-950-VIRTUAL active for online subscriptions."
        ),
        initial_state={
            "cards": {
                "CARD-950-DEBIT": {"user_id": "U-950", "status": "active"},
                "CARD-950-TRAVEL": {"user_id": "U-950", "status": "active"},
                "CARD-950-VIRTUAL": {"user_id": "U-950", "status": "active"},
            }
        },
        success=[
            {"path": ["cards", "CARD-950-DEBIT", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-950-TRAVEL", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-950-VIRTUAL", "status"], "equals": "active"},
        ],
        allowed_tools=["freeze_card", "finish"],
        max_turns=7,
    ),
    ToolDialogTask(
        domain="banking",
        task_id="hard-freeze-primary-not-secondary",
        instruction=(
            "User U-960 says only the primary card was stolen. Freeze CARD-960-PRIMARY. "
            "Keep CARD-960-SPOUSE and CARD-960-SAVINGS active."
        ),
        initial_state={
            "cards": {
                "CARD-960-PRIMARY": {"user_id": "U-960", "status": "active"},
                "CARD-960-SPOUSE": {"user_id": "U-960", "status": "active"},
                "CARD-960-SAVINGS": {"user_id": "U-960", "status": "active"},
            }
        },
        success=[
            {"path": ["cards", "CARD-960-PRIMARY", "status"], "equals": "frozen"},
            {"path": ["cards", "CARD-960-SPOUSE", "status"], "equals": "active"},
            {"path": ["cards", "CARD-960-SAVINGS", "status"], "equals": "active"},
        ],
        allowed_tools=["freeze_card", "finish"],
        max_turns=6,
    ),
)


def _generated_retail_tasks(count: int) -> list[ToolDialogTask]:
    tasks: list[ToolDialogTask] = []
    for i in range(count):
        n = 1000 + i
        if i % 2 == 0:
            order_id = f"O-{n}"
            other_id = f"O-{n + 5000}"
            tasks.append(
                ToolDialogTask(
                    domain="retail",
                    task_id=f"generated-cancel-{i + 1:03d}",
                    instruction=f"Customer C-{n} wants duplicate order {order_id} cancelled. Do not alter {other_id}.",
                    initial_state={
                        "orders": {
                            order_id: {
                                "customer_id": f"C-{n}",
                                "status": "processing" if i % 4 else "packed",
                                "address": f"{10 + i} Willow St",
                            },
                            other_id: {
                                "customer_id": f"C-{n}",
                                "status": "processing",
                                "address": f"{10 + i} Willow St",
                            },
                        }
                    },
                    success=[
                        {"path": ["orders", order_id, "status"], "equals": "cancelled"},
                        {"path": ["orders", other_id, "status"], "equals": "processing"},
                    ],
                    allowed_tools=["cancel_order", "finish"],
                )
            )
        else:
            order_id = f"O-{n}"
            address = f"{100 + i} Cedar Way, Unit {(i % 9) + 1}"
            tasks.append(
                ToolDialogTask(
                    domain="retail",
                    task_id=f"generated-address-{i + 1:03d}",
                    instruction=f"Update order {order_id} shipping address to '{address}' before it ships.",
                    initial_state={
                        "orders": {
                            order_id: {
                                "customer_id": f"C-{n}",
                                "status": "processing",
                                "address": f"{100 + i} Cedar Way",
                            }
                        }
                    },
                    success=[{"path": ["orders", order_id, "address"], "equals": address}],
                    allowed_tools=["update_shipping_address", "finish"],
                )
            )
    return tasks


def _generated_airline_tasks(count: int) -> list[ToolDialogTask]:
    seats = ["A", "B", "C", "D", "E", "F"]
    tasks: list[ToolDialogTask] = []
    for i in range(count):
        n = 2000 + i
        reservation_id = f"R-{n}"
        seat = f"{2 + (i % 28)}{seats[i % len(seats)]}"
        tasks.append(
            ToolDialogTask(
                domain="airline",
                task_id=f"generated-seat-{i + 1:03d}",
                instruction=f"Assign passenger P-{n} seat {seat} on active reservation {reservation_id}.",
                initial_state={
                    "reservations": {
                        reservation_id: {
                            "passenger_id": f"P-{n}",
                            "status": "active",
                            "seat": None if i % 3 == 0 else f"{3 + (i % 20)}C",
                        },
                        f"R-{n + 5000}": {
                            "passenger_id": f"P-{n + 5000}",
                            "status": "cancelled",
                            "seat": None,
                        },
                    }
                },
                success=[{"path": ["reservations", reservation_id, "seat"], "equals": seat}],
                allowed_tools=["assign_seat", "finish"],
            )
        )
    return tasks


def _generated_banking_tasks(count: int) -> list[ToolDialogTask]:
    reasons = ["lost-wallet report", "fraud alert", "travel theft report", "merchant compromise"]
    tasks: list[ToolDialogTask] = []
    for i in range(count):
        n = 3000 + i
        card_id = f"CARD-{n}"
        other_id = f"CARD-{n + 5000}"
        tasks.append(
            ToolDialogTask(
                domain="banking",
                task_id=f"generated-freeze-{i + 1:03d}",
                instruction=f"Freeze {card_id} for user U-{n} after a {reasons[i % len(reasons)]}. Leave {other_id} active.",
                initial_state={
                    "cards": {
                        card_id: {"user_id": f"U-{n}", "status": "active"},
                        other_id: {"user_id": f"U-{n}", "status": "active"},
                    }
                },
                success=[
                    {"path": ["cards", card_id, "status"], "equals": "frozen"},
                    {"path": ["cards", other_id, "status"], "equals": "active"},
                ],
                allowed_tools=["freeze_card", "finish"],
            )
        )
    return tasks


def _build_tasks() -> tuple[ToolDialogTask, ...]:
    target_per_domain = 50
    counts = {
        "retail": sum(1 for task in _SEED_TASKS if task.domain == "retail"),
        "airline": sum(1 for task in _SEED_TASKS if task.domain == "airline"),
        "banking": sum(1 for task in _SEED_TASKS if task.domain == "banking"),
    }
    generated = [
        *_generated_retail_tasks(target_per_domain - counts["retail"]),
        *_generated_airline_tasks(target_per_domain - counts["airline"]),
        *_generated_banking_tasks(target_per_domain - counts["banking"]),
    ]
    return (*_SEED_TASKS, *generated, *_HARD_TASKS)


TASKS: tuple[ToolDialogTask, ...] = _build_tasks()


def tools_for(task: ToolDialogTask) -> list[dict[str, Any]]:
    allowed = set(task.allowed_tools)
    return [tool for tool in COMMON_TOOLS if tool["function"]["name"] in allowed]


def task_spec(task: ToolDialogTask) -> TaskSpec:
    task_id = f"{SOURCE_NAME}__{task.domain}__{task.task_id}"
    group = f"{SOURCE_NAME}/{task.domain}/{task.task_id}"
    return TaskSpec(
        task_id=task_id,
        capability="tool_dialogue",
        source=SourceRef(
            name=SOURCE_NAME,
            version=SOURCE_VERSION,
            policy="train_allowed",
            url_or_ref="generated://tau_custom",
        ),
        input=TaskInput(
            messages=[{"role": "user", "content": task.instruction}],
            tools=tools_for(task),
        ),
        environment=EnvironmentSpec(
            harness="tool_dialog",
            wall_time_seconds=300,
        ),
        grader=GraderSpec(
            type="db_state",
            expected_answer={
                "domain": task.domain,
                "initial_state": task.initial_state,
                "success": task.success,
                "allowed_tools": task.allowed_tools,
                "max_turns": task.max_turns,
            },
            success_threshold=1.0,
        ),
        splitting=SplittingSpec(
            group_id=group,
            split="grpo_train",
            contamination_group=group,
        ),
        metadata=TaskMetadata(
            domain=task.domain,
            subdomain="custom_tau",
            tags=[
                "tool_dialogue",
                "tau_custom",
                "generated",
                *(["hard", "multi_step"] if task.task_id.startswith("hard-") else []),
            ],
            requires_tools=True,
            estimated_worker_calls=task.max_turns,
        ),
    )


def materialize_tool_dialog_tasks(
    *,
    out_jsonl: Path,
    report_out: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    selected = list(TASKS[:limit] if limit is not None else TASKS)
    specs = [task_spec(task) for task in selected]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for spec in specs:
            f.write(json.dumps(spec.model_dump(mode="json"), sort_keys=True) + "\n")

    report = {
        "version": "tau_custom_tool_dialog_tasks_v1",
        "source": SOURCE_NAME,
        "task_count": len(specs),
        "out_jsonl": str(out_jsonl),
        "splits": sorted({spec.splitting.split for spec in specs}),
        "domains": sorted({str(spec.metadata.domain) for spec in specs}),
        "live_calls": False,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
