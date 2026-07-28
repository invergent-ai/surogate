"""Child process for the ``sql_exec`` grader: run one query, print a digest.

Runs in its own process so a runaway join cannot hang or OOM the training
loop: the parent enforces a wall-clock timeout, and this process caps its
own address space. The connection is opened read-only, so a candidate query
cannot mutate the benchmark database.

Prints one JSON line: ``{"ok": true, "n": <distinct rows>, "hash": <sha1>,
"reads_table": <bool>}`` or ``{"ok": false, "error": "..."}``. The digest is
over the SORTED SET of row tuples, matching BIRD's official execution-accuracy
comparison (``set(predicted) == set(gold)``), including its type semantics —
values are not normalized, so our numbers stay comparable with published ones.

``reads_table`` comes from sqlite's authorizer callback (exact, not parsed
from query text) and exists to close a reward-hacking surface: a constant
query like ``SELECT 1`` matches any gold whose answer happens to be 1, so the
grader requires that a candidate actually read the database.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sqlite3
import sys

MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def digest(rows: list[tuple]) -> tuple[int, str]:
    distinct = sorted({tuple(row) for row in rows}, key=repr)
    sha = hashlib.sha1()
    for row in distinct:
        sha.update(repr(row).encode("utf-8", "replace"))
        sha.update(b"\x1e")
    return len(distinct), sha.hexdigest()


def main(argv: list[str]) -> int:
    db_path, sql = argv[1], argv[2]
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ValueError, OSError):
        pass  # limit already lower, or unsupported — parent timeout still applies
    read_tables: set[str] = set()

    def authorizer(action, arg1, _arg2, _dbname, _source):
        if action == sqlite3.SQLITE_READ and arg1:
            read_tables.add(str(arg1))
        return sqlite3.SQLITE_OK

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            con.set_authorizer(authorizer)
            rows = con.execute(sql).fetchall()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 — any failure is a zero-reward answer
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}))
        return 0
    n, sha = digest(rows)
    print(json.dumps({"ok": True, "n": n, "hash": sha,
                      "reads_table": bool(read_tables)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
