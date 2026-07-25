"""Multi-hop tool-use environment for turn-level GRPO/OPD diagnostics.

A synthetic knowledge graph is generated deterministically from a seed. Each
question requires chaining ``k`` lookups to reach the answer:

    "Which city is the headquarters of the maker of Zorbex?"
      -> lookup: Zorbex        (product)  -> made_by: Kelmoor Industries
      -> lookup: Kelmoor Industries       -> led_by:  Prisa Vantel
      -> lookup: Prisa Vantel             -> based_in: Threnholm

The model drives the interaction with a plain-text tool protocol (no
tool-call parser required, so it works with any chat model):

    <tool>lookup: Kelmoor Industries</tool>
    <answer>Threnholm</answer>

This is the cheapest faithful analogue of the paper's Multi-Hop Search task:
turn count varies with hop depth, failures degenerate into repeated or
invalid lookups, and successes stay information-rich. That is exactly the
structure the turn-level diagnostics need — a spread of trajectory depths
with a clean success/failure split at every depth.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset as HFDataset

# ── Synthetic world ──────────────────────────────────────────────────────────

_SYL_A = ["kel", "zor", "thren", "mar", "vand", "pris", "gal", "orm", "brann", "quil"]
_SYL_B = ["mo", "be", "hol", "ti", "es", "van", "dro", "wy", "ar", "ovi"]
_SYL_C = ["ra", "nex", "th", "lum", "dar", "sk", "vel", "on", "isk", "wen"]
_SUFFIX_CO = ["Industries", "Dynamics", "Systems", "Collective", "Works"]

# Relation chain: product -> company -> person -> city -> region
_CHAIN = [
    ("product", "made_by", "company"),
    ("company", "led_by", "person"),
    ("person", "based_in", "city"),
    ("city", "located_in", "region"),
]

_REL_PHRASE = {
    "made_by": "the maker of",
    "led_by": "the head of",
    "based_in": "the home city of",
    "located_in": "the region containing",
}

# Distractor attributes so a single lookup never leaks the answer.
_FILLER = {
    "product": ["category", "launch_year", "unit_price"],
    "company": ["founded", "employees", "sector"],
    "person": ["role", "tenure_years"],
    "city": ["population", "climate"],
    "region": ["area_km2"],
}


def _name_pool(rng: random.Random) -> list[str]:
    """All syllable combinations, shuffled. Deterministic and collision-free."""
    pool = [(a + b + c).capitalize() for a in _SYL_A for b in _SYL_B for c in _SYL_C]
    rng.shuffle(pool)
    return pool


def _build_world(rng: random.Random, n_chains: int) -> dict[str, dict[str, Any]]:
    """Build entity table: name -> {kind, <relation>: target, <fillers>}."""
    world: dict[str, dict[str, Any]] = {}
    pool = _name_pool(rng)
    needed = n_chains * 5
    if needed > len(pool):
        raise ValueError(f"n_chains={n_chains} needs {needed} names but the pool holds {len(pool)}")
    cursor = 0

    def fresh(suffix: str = "") -> str:
        nonlocal cursor
        nm = pool[cursor]
        cursor += 1
        return nm + (f" {suffix}" if suffix else "")

    for _ in range(n_chains):
        # One full chain: product -> company -> person -> city -> region
        names = {
            "product": fresh(),
            "company": fresh(rng.choice(_SUFFIX_CO)),
            "person": fresh(),
            "city": fresh(),
            "region": fresh(),
        }
        for kind, rel, tgt_kind in _CHAIN:
            ent: dict[str, Any] = {"kind": kind, rel: names[tgt_kind]}
            for f in _FILLER[kind]:
                ent[f] = _filler_value(rng, f)
            world[names[kind]] = ent
        region: dict[str, Any] = {"kind": "region"}
        for f in _FILLER["region"]:
            region[f] = _filler_value(rng, f)
        world[names["region"]] = region

    return world


def _filler_value(rng: random.Random, field: str) -> Any:
    if field in ("launch_year", "founded"):
        return rng.randint(1950, 2020)
    if field in ("employees", "population"):
        return rng.randint(200, 900_000)
    if field == "unit_price":
        return round(rng.uniform(5, 500), 2)
    if field == "tenure_years":
        return rng.randint(1, 25)
    if field == "area_km2":
        return rng.randint(500, 90_000)
    if field == "category":
        return rng.choice(["tooling", "sensor", "polymer", "reagent"])
    if field == "sector":
        return rng.choice(["industrial", "biotech", "aerospace", "materials"])
    if field == "role":
        return rng.choice(["chief executive", "managing director", "president"])
    if field == "climate":
        return rng.choice(["temperate", "arid", "humid", "alpine"])
    return "n/a"


def _make_question(world: dict[str, dict[str, Any]], start: str, hops: int) -> tuple[str, str]:
    """Walk `hops` relations from `start`; return (question text, gold answer)."""
    cur = start
    rels: list[str] = []
    for _ in range(hops):
        ent = world[cur]
        rel = next(r for r in ent if r in _REL_PHRASE)
        rels.append(rel)
        cur = ent[rel]

    # Wrap outward in walk order, so the outermost phrase is the last hop:
    # rels=[made_by, led_by, based_in] -> "the home city of the head of the maker of X"
    phrase = start
    for rel in rels:
        phrase = f"{_REL_PHRASE[rel]} {phrase}"
    return f"What is {phrase}?", cur


SYSTEM_PROMPT = """You answer multi-hop questions by looking entities up one at a time.

You have exactly one tool. To use it, emit ONLY this, nothing else:
<tool>lookup: EXACT ENTITY NAME</tool>

The result is returned as JSON. Follow the relation fields (made_by, led_by,
based_in, located_in) to walk toward the answer. Look up one entity per turn.

When you know the answer, emit ONLY:
<answer>THE ANSWER</answer>

Rules:
- One <tool> OR one <answer> per message. Never both.
- Entity names must be copied exactly as they appear.
- Do not guess an answer you have not reached by lookup."""


_TOOL_RE = re.compile(r"<tool>\s*lookup:\s*(.+?)\s*</tool>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _last_text(messages) -> str:
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "assistant":
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            if isinstance(content, list):  # content parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content or ""
    return ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


class MultiHopToolEnv(vf.MultiTurnEnv):
    """Text-protocol lookup loop over a synthetic knowledge graph.

    The world is held on the env, not in per-example ``info`` — it is shared by
    every example and duplicating it per row would bloat the Arrow table.
    """

    def __init__(self, world: dict[str, dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.world = world

    async def env_response(self, messages, state, **kwargs):
        text = _last_text(messages)

        ans = _ANSWER_RE.search(text)
        if ans:
            state["submitted_answer"] = ans.group(1).strip()
            # Terminal: no further model turn.
            state["final_env_response"] = [{"role": "user", "content": "Answer recorded."}]
            return state["final_env_response"]

        world = self.world
        call = _TOOL_RE.search(text)
        if not call:
            state.setdefault("malformed_turns", 0)
            state["malformed_turns"] += 1
            return [
                {
                    "role": "user",
                    "content": "Malformed. Emit exactly one <tool>lookup: NAME</tool> or one <answer>...</answer>.",
                }
            ]

        entity = call.group(1).strip()
        state.setdefault("lookups", []).append(entity)

        record = world.get(entity)
        if record is None:
            # Case-insensitive rescue before declaring it unknown.
            match = next((k for k in world if _norm(k) == _norm(entity)), None)
            record = world.get(match) if match else None

        if record is None:
            return [{"role": "user", "content": f'{{"error": "no entity named {entity!r}"}}'}]
        return [{"role": "user", "content": json.dumps(record, sort_keys=True)}]

    @vf.stop
    async def answer_submitted(self, state) -> bool:
        return state.get("submitted_answer") is not None


# ── Rewards ──────────────────────────────────────────────────────────────────


def correctness_reward(state, answer, **kwargs) -> float:
    """1.0 iff the submitted answer matches the gold entity."""
    submitted = state.get("submitted_answer")
    if submitted is None:
        return 0.0
    gold = answer if isinstance(answer, str) else answer.get("gold", "")
    return 1.0 if _norm(submitted) == _norm(gold) else 0.0


def load_environment(**kwargs) -> vf.Environment:
    """Entry point for Surogate GRPO."""
    seed = int(kwargs.pop("seed", 0))
    num_examples = int(kwargs.pop("num_examples", 2000))
    n_chains = int(kwargs.pop("n_chains", 120))
    min_hops = int(kwargs.pop("min_hops", 2))
    max_hops = int(kwargs.pop("max_hops", 4))
    max_turns = int(kwargs.pop("max_turns", 12))

    rng = random.Random(seed)
    world = _build_world(rng, n_chains)
    products = [n for n, e in world.items() if e["kind"] == "product"]

    questions: list[str] = []
    answers: list[str] = []
    infos: list[dict] = []
    for i in range(num_examples):
        start = products[i % len(products)]
        hops = rng.randint(min_hops, max_hops)
        q, gold = _make_question(world, start, hops)
        questions.append(q)
        answers.append(gold)
        infos.append({"hops": hops, "start": start})

    env_dataset = HFDataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "task": ["multihop-tools"] * len(questions),
            "info": infos,
        }
    )

    rubric = vf.Rubric(funcs=[correctness_reward], weights=[1.0])

    return MultiHopToolEnv(
        world=world,
        dataset=env_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )
