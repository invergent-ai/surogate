"""Measure finite-choice classification latency against a running native server."""

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def run(url, model, repeats=5):
    results = []
    for count, collisions in [(4, False), (16, False), (32, False), (8, True)]:
        properties = {
            f"flag_{i}": ({"type": "string", "enum": ["not applicable", "not available", "not authorized", "approved"]}
                          if collisions else {"type": "boolean"})
            for i in range(count)
        }
        schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
        samples = []
        for trial in range(repeats + 1):
            body = {
                "model": model,
                "messages": [{"role": "user", "content": f"Trial {trial}. Classify this support request: the production server is down. "
                              + "The customer wants help. " * 80}],
                "parallel_decoding": True, "temperature": 1, "max_tokens": 600,
                "response_format": {"type": "json_schema", "json_schema": {"name": "triage", "schema": schema}},
            }
            started = time.perf_counter()
            response = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=120)
            elapsed = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            content = json.loads(response.json()["choices"][0]["message"]["content"])
            if set(content) != set(properties):
                raise ValueError("classifier returned unexpected fields")
            if trial:
                samples.append(elapsed)
        results.append({"fields": count, "collisions": collisions,
                        "median_ms": statistics.median(samples), "samples_ms": samples})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = json.dumps(run(args.url, args.model, args.repeats), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
