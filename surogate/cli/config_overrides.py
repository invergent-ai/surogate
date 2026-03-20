from __future__ import annotations

from typing import Any

import yaml


def _normalize_key(key: str) -> str:
    # Accept both kebab-case and snake_case in CLI keys.
    return ".".join(part.replace("-", "_") for part in key.split("."))


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    cur = target
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def parse_cli_overrides(unknown_args: list[str]) -> dict[str, Any]:
    """Parse unknown CLI args into nested config overrides.

    Supported forms:
      --key=value
      --nested.key=value
      --flag                -> true
      --no-flag             -> false
      --key value
    """
    overrides: dict[str, Any] = {}

    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected CLI token (expected --key): {token}")

        body = token[2:]
        if not body:
            raise ValueError("Empty CLI override key")

        if "=" in body:
            key_raw, value_raw = body.split("=", 1)
        else:
            if body.startswith("no-"):
                key_raw = body[3:]
                value_raw = "false"
            elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                key_raw = body
                i += 1
                value_raw = unknown_args[i]
            else:
                key_raw = body
                value_raw = "true"

        key = _normalize_key(key_raw.strip())
        if not key:
            raise ValueError("Empty CLI override key")
        path = [p for p in key.split(".") if p]
        if not path:
            raise ValueError(f"Invalid CLI override key: {key_raw}")

        _set_nested(overrides, path, _parse_scalar(value_raw))
        i += 1

    return overrides


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base and return base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base

