"""Probe a served model against the multihop-tools protocol.

Answers the question the reward number cannot: when reward is 0, is the model
failing to *follow the protocol* (never emits a well-formed <tool> call) or
failing to *reason* (walks the graph wrong / gives up)? Those need different
fixes, so measure before changing anything.

Usage:
  python examples/turnopd/probe_env.py --port 8007 --model Qwen/Qwen3.5-0.8B -n 20
"""

import argparse
import json
import re
import sys
import urllib.request

sys.path.insert(0, "environments/multihop-tools")
import multihop_tools as M  # noqa: E402


def chat(port, model, messages, max_tokens=256, temperature=1.0):
    body = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--show", type=int, default=2, help="print this many full transcripts")
    args = ap.parse_args()

    env = M.load_environment(num_examples=args.n, n_chains=120, seed=0)
    world = env.world

    stats = {
        "solved": 0,
        "gave_answer": 0,
        "wrong_answer": 0,
        "any_valid_tool": 0,
        "never_valid_tool": 0,
        "hit_turn_cap": 0,
        "unknown_entity_calls": 0,
        "malformed_turns": 0,
        "total_turns": 0,
        "repeat_lookups": 0,
    }
    first_action = {"tool": 0, "answer": 0, "malformed": 0}

    for i, row in enumerate(list(env.dataset)[: args.n]):
        messages = [
            {"role": "system", "content": M.SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        seen_lookups, valid_tool, answered = [], False, None

        for turn in range(args.max_turns):
            text = chat(args.port, args.model, messages, temperature=args.temperature)
            messages.append({"role": "assistant", "content": text})
            stats["total_turns"] += 1

            ans = M._ANSWER_RE.search(text)
            call = M._TOOL_RE.search(text)

            if turn == 0:
                first_action["answer" if ans else ("tool" if call else "malformed")] += 1

            if ans:
                answered = ans.group(1).strip()
                break
            if not call:
                stats["malformed_turns"] += 1
                messages.append({"role": "user", "content": "Malformed. Emit one <tool>lookup: NAME</tool> or <answer>...</answer>."})
                continue

            entity = call.group(1).strip()
            if entity in seen_lookups:
                stats["repeat_lookups"] += 1
            seen_lookups.append(entity)

            rec = world.get(entity) or next((world[k] for k in world if M._norm(k) == M._norm(entity)), None)
            if rec is None:
                stats["unknown_entity_calls"] += 1
                messages.append({"role": "user", "content": f'{{"error": "no entity named {entity!r}"}}'})
            else:
                valid_tool = True
                messages.append({"role": "user", "content": json.dumps(rec, sort_keys=True)})
        else:
            stats["hit_turn_cap"] += 1

        if valid_tool:
            stats["any_valid_tool"] += 1
        else:
            stats["never_valid_tool"] += 1
        if answered is not None:
            stats["gave_answer"] += 1
            if M._norm(answered) == M._norm(row["answer"]):
                stats["solved"] += 1
            else:
                stats["wrong_answer"] += 1

        if i < args.show:
            print(f"\n{'=' * 78}\nTRANSCRIPT {i}  (hops={row['info']['hops']}, gold={row['answer']})")
            for m in messages:
                if m["role"] == "system":
                    continue
                print(f"  [{m['role']:9}] {m['content'][:150].replace(chr(10), ' ')}")
            print(f"  -> answered={answered!r} correct={answered is not None and M._norm(answered) == M._norm(row['answer'])}")

    n = args.n
    print(f"\n{'=' * 78}\nPROBE: {args.model} (n={n}, temp={args.temperature})\n{'=' * 78}")
    print(f"  solved                {stats['solved']:4d} / {n}  ({stats['solved'] / n:.0%})")
    print(f"  gave an answer        {stats['gave_answer']:4d} / {n}   (wrong: {stats['wrong_answer']})")
    print(f"  made >=1 VALID lookup {stats['any_valid_tool']:4d} / {n}")
    print(f"  never a valid lookup  {stats['never_valid_tool']:4d} / {n}")
    print(f"  hit turn cap          {stats['hit_turn_cap']:4d} / {n}")
    print(f"  first action: tool={first_action['tool']} answer={first_action['answer']} malformed={first_action['malformed']}")
    print(f"  malformed turns       {stats['malformed_turns']:4d} / {stats['total_turns']} turns")
    print(f"  unknown-entity calls  {stats['unknown_entity_calls']:4d}")
    print(f"  repeated lookups      {stats['repeat_lookups']:4d}")
    print("\n  VERDICT: ", end="")
    if stats["any_valid_tool"] == 0:
        print("PROTOCOL failure — never produced a usable tool call.")
    elif stats["gave_answer"] == 0:
        print("TERMINATION failure — tools work, but it never commits to an answer.")
    elif stats["solved"] == 0:
        print("REASONING failure — follows protocol and answers, but always wrong.")
    else:
        print(f"partial competence — {stats['solved'] / n:.0%} solve rate.")


if __name__ == "__main__":
    main()
