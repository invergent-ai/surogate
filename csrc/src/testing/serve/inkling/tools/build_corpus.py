# agent-EN-f2 (ENGINE-INKLING lane f2), 2026-11-21.
# Builds the renderer corpus: OpenAI chat bodies as agent-IK's gateway sent them to vLLM for Inkling-2-Mini, rebuilt
# from the REAL surogates sessions of agent-IK's reads (read only: /mnt/data/work/rl/inkling-2026-11-21).
#
# What is real and what is not (the gateway ran RECORD_BODY=0, so no request body was recorded):
#  - conversation = the harness's own replay (surogates.harness.loop_context_replay._rebuild_messages over the session's
#    events.jsonl up to the i-th llm.request) + the harness's wire step (surogates.harness.loop._to_api_messages: reasoning
#    -> reasoning_content, the '(no reasoning recorded)' placeholder). Real user text, real Inkling reasoning / tool calls
#    / tool results. Checked per call against the gateway record's n_messages (field "n_messages_match").
#  - title calls (sys_sha fea53892ede7) = surogates.harness.title_generator's exact body (system text hash matches).
#  - main-loop system prompt = NOT recorded: composed from the harness's own prompt fragments (prompts/identity,
#    guidance) -- real harness text, not the exact recorded system message (record sys_sha kept in meta for reference).
#  - tools = NOT recorded: the harness's real builtin tool schemas (ToolRegistry.get_schemas of surogates/tools/builtin,
#    dumped by dump_tools.py), the count matched to the record's n_tools where possible.
#  - gateway step (eff_gateway_ik.py, surogates path): setdefault PIN (T 1.0, top_p 1.0, top_k -1), chat_template_kwargs
#    {enable_thinking: True, preserve_thinking: True} under the request's own, model "cand", stream False.
# Run with the surogates venv: /mnt/data/work/rl/surogates/.venv/bin/python -I build_corpus.py OUT.jsonl
import hashlib
import json
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, "/mnt/data/work/repos/surogates")
import surogates.harness.loop as L  # noqa: E402
import surogates.harness.loop_context_replay as R  # noqa: E402
import surogates.harness.title_generator as T  # noqa: E402

IK = "/mnt/data/work/rl/inkling-2026-11-21"
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = "/mnt/data/work/repos/surogates/surogates/harness/prompts"
PIN = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
CTK = {"enable_thinking": True, "preserve_thinking": True}
TITLE_SHA = "fea53892ede7"

# (label, glob root of task dirs, calls dirs, sessions to take, call points per session)
SOURCES = [
    ("ab", [f"{IK}/w2/run1/ab-c-s0/r1/tasks", f"{IK}/w2/run1/ab-c-s1/r2/tasks"], [f"{IK}/w2/run1/calls"], 36, 2),
    ("eog", [f"{IK}/w2/run1/eog-c/r1/tasks", f"{IK}/w2/run1/eog-c/r2/tasks"], [f"{IK}/w2/run1/calls"], 26, 2),
    ("ifeval", [f"{IK}/w2/if2/ifeval-c/r1/tasks"], [f"{IK}/w2/if2/calls_p2", f"{IK}/w2/run1/calls_p2"], 14, 1),
    ("hox", [f"{IK}/hx/hox1/hox/r1/tasks", f"{IK}/hx/hox1/hox/r2/tasks"], [f"{IK}/hx/hox1/calls"], 26, 2),
    ("office", [f"{IK}/ow/ho60/p1"], [f"{IK}/ow/ho60/calls"], 26, 2),
    ("claw", [f"{IK}/dv/p3/claw-c"], None, 6, 1),
    ("ifbench", [f"{IK}/dv/p3/ifbench-c"], None, 5, 1),
    ("dab", [f"{IK}/dv/p3/dab-c"], None, 3, 1),
    ("oqa", [f"{IK}/dv/p3/oqa-c"], None, 3, 1),
    ("office156", [f"{IK}/dv/p3/office-c"], None, 4, 1),
    ("wsb", [f"{IK}/dv/p3/wsb-c"], None, 2, 1),
]
MAX_PROMPT_CHARS = 700_000  # ~170k tokens: keep vectors under the 262k serving context


def fragment(path):
    text = open(os.path.join(PROMPTS, path)).read()
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    return text.strip()


SYSTEM_PROMPT = "\n\n".join(
    fragment(p)
    for p in (
        "identity/default_personality.md",
        "identity/workspace_rules.md",
        "guidance/working_principles.md",
        "guidance/parallel_tools.md",
    )
)
BUILTIN_TOOLS = json.load(open(os.path.join(HERE, "surogates_builtin_tools.json")))
BUILTIN_BY_NAME = {t["function"]["name"]: t for t in BUILTIN_TOOLS}


def find_events(roots):
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "events.jsonl" in files:
                out.append(dirpath)
    return sorted(out)


def load_events(path):
    events = []
    for line in open(os.path.join(path, "events.jsonl")):
        if line.strip():
            r = json.loads(line)
            events.append(SimpleNamespace(id=r.get("id"), type=r.get("type"), data=r.get("data") or {}))
    return events


def session_id(path, events):
    meta = os.path.join(path, "meta.json")
    if os.path.exists(meta):
        try:
            sid = json.load(open(meta)).get("session_id")
            if sid:
                return sid
        except Exception:
            pass
    return None


def call_records(calls_dirs, sid):
    if not calls_dirs or not sid:
        return []
    for d in calls_dirs:
        f = os.path.join(d, f"srs-{sid}.jsonl")
        if os.path.exists(f):
            return [json.loads(x) for x in open(f) if x.strip()]
    return []


def tool_subset(conv, n_tools):
    """Real builtin schemas: the ones the conversation called first, then the registry order, n_tools of them."""
    called = []
    for m in conv:
        for tc in m.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if name in BUILTIN_BY_NAME and name not in called:
                called.append(name)
    names = called + [t["function"]["name"] for t in BUILTIN_TOOLS if t["function"]["name"] not in called]
    n = max(1, min(n_tools or 26, len(names)))
    return [BUILTIN_BY_NAME[x] for x in names[:n]]


def gateway(body):
    for k, v in PIN.items():
        body.setdefault(k, v)
    body["chat_template_kwargs"] = {**CTK, **(body.get("chat_template_kwargs") or {})}
    body["model"] = "cand"
    body["stream"] = False
    body.pop("stream_options", None)
    return body


def main():
    out_path = sys.argv[1]
    items = []
    seen_title = 0
    for label, roots, calls_dirs, n_sessions, n_points in SOURCES:
        paths = find_events(roots)
        if not paths:
            print(f"{label}: no sessions", file=sys.stderr)
            continue
        step = max(1, len(paths) // n_sessions)
        chosen = paths[::step][:n_sessions]
        n_items = 0
        for path in chosen:
            events = load_events(path)
            sid = session_id(path, events)
            recs = call_records(calls_dirs, sid)
            main_recs = [r for r in recs if r.get("sys_sha") != TITLE_SHA and r.get("status") == 200]
            req_idx = [i for i, e in enumerate(events) if e.type == "llm.request"]
            if not req_idx:
                continue
            wires = []
            for k, idx in enumerate(req_idx):
                conv = R.ContextReplayMixin._rebuild_messages(None, events[:idx])
                wire = L._to_api_messages(conv)
                wires.append((k, wire))
            # call points: first, a middle one, the last; prefer a call that carries the placeholder
            ks = sorted({0, len(wires) // 2, len(wires) - 1})
            placeholder = [k for k, w in wires if any(m.get("reasoning_content") == "(no reasoning recorded)" for m in w)]
            picks = []
            if placeholder:
                picks.append(placeholder[len(placeholder) // 2])
            for k in reversed(ks):
                if len(picks) >= n_points:
                    break
                if k not in picks:
                    picks.append(k)
            for k in sorted(picks):
                wire = wires[k][1]
                rec = main_recs[k] if k < len(main_recs) else {}
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + wire
                body = {"model": "surogate/Surogate3.7-35B-A3B", "messages": messages,
                        "tools": tool_subset(wire, rec.get("n_tools")), "stream": True}
                body = gateway(body)
                if len(json.dumps(body, ensure_ascii=False)) > MAX_PROMPT_CHARS:
                    continue
                n_ph = sum(1 for m in wire if m.get("reasoning_content") == "(no reasoning recorded)")
                items.append({
                    "case": f"{label}-{os.path.basename(path)}-{sid[:8] if sid else 'x'}-c{k:03d}",
                    "source": os.path.relpath(path, IK),
                    "body": body,
                    "meta": {
                        "kind": "main", "call_index": k, "n_calls": len(wires), "placeholders": n_ph,
                        "n_messages": len(messages), "record_n_messages": rec.get("n_messages"),
                        "n_messages_match": (rec.get("n_messages") == len(messages)) if rec else None,
                        "record_n_tools": rec.get("n_tools"), "record_sys_sha": rec.get("sys_sha"),
                        "system_prompt": "harness fragments (not the recorded text)",
                    },
                })
                n_items += 1
            # the session's title call, exactly as title_generator builds it (first sessions of each source)
            first_user = next((e.data.get("content") for e in events if e.type == "user.message"), None)
            if first_user and seen_title < 24 and isinstance(first_user, str):
                snippet = first_user[:500]
                body = {"model": "surogate/Surogate3.7-35B-A3B",
                        "messages": [{"role": "system", "content": T._TITLE_PROMPT},
                                     {"role": "user", "content": T._TITLE_USER_TEMPLATE.format(snippet=snippet)}],
                        "max_tokens": 32, "temperature": 0.3, "stream": False,
                        "chat_template_kwargs": {"enable_thinking": False}}
                sha = hashlib.sha1(T._TITLE_PROMPT.encode()).hexdigest()[:12]
                items.append({"case": f"title-{label}-{sid[:8] if sid else os.path.basename(path)}",
                              "source": os.path.relpath(path, IK), "body": gateway(body),
                              "meta": {"kind": "title", "sys_sha": sha, "sys_sha_match": sha == TITLE_SHA}})
                seen_title += 1
        print(f"{label}: {len(paths)} sessions, chose {len(chosen)}, {n_items} main items", file=sys.stderr)
    with open(out_path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    main = [i for i in items if i["meta"]["kind"] == "main"]
    match = sum(1 for i in main if i["meta"]["n_messages_match"])
    known = sum(1 for i in main if i["meta"]["n_messages_match"] is not None)
    print(f"total {len(items)} items ({len(main)} main, n_messages match {match}/{known} with a record; "
          f"{sum(1 for i in main if i['meta']['placeholders'])} with placeholders)", file=sys.stderr)


if __name__ == "__main__":
    main()
