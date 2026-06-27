"""Top-level ``director`` entrypoint."""

from __future__ import annotations

import argparse

from .agentic import run as agentic_run
from .fugu import run as fugu_run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="director", description="Fugu LLM orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)
    fugu_run.add_subparsers(sub)
    agentic_run.add_subparsers(sub)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
