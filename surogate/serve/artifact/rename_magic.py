"""Rewrite a serving artifact's magic from the pre-rename name, in place.

The artifact format is `magic(8) | json_bytes(8) | JSON directory | payload`, and
nothing checksums the header — the magic is a pure identifier. The word appears
exactly once in a whole artifact: those first six bytes. The JSON directory holds
`model_id` and `weights_id`, not the format's name, and no payload encodes it.

So renaming the project does not require reconverting anything. This patches the
six bytes and renames the file, which takes milliseconds on an artifact of any
size — a 111 GB Flash-Next artifact included.

The structure is untouched, so the version stays 2: nothing about the layout
changed and claiming a new version would misdescribe the file.

    python -m surogate.serve.artifact.rename_magic ARTIFACT [ARTIFACT ...]
      --dry-run     report what would change and touch nothing
      --keep-name   patch the magic but leave the .ninfer filename alone
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: What a pre-rename artifact starts with, and what it becomes. Same version:
#: only the identifier changed.
LEGACY_MAGIC = b"NINFER\x00\x02"
CURRENT_MAGIC = b"SINFER\x00\x02"
#: v1 artifacts were already unsupported before the rename; they are detected so
#: the message can say so rather than "not a serving artifact".
LEGACY_V1_MAGIC = b"NINFER\x00\x01"

LEGACY_SUFFIX = ".ninfer"
CURRENT_SUFFIX = ".sinfer"


def inspect(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(8)


def migrate(path: Path, *, dry_run: bool, keep_name: bool) -> bool:
    """Patch one artifact. Returns True when it changed (or would have)."""

    magic = inspect(path)
    if magic == CURRENT_MAGIC:
        target = path.with_suffix(CURRENT_SUFFIX)
        if not keep_name and path.suffix == LEGACY_SUFFIX:
            if not dry_run:
                path.rename(target)
            print(f"{path}: magic already current; {'would rename' if dry_run else 'renamed'} "
                  f"to {target.name}")
            return True
        print(f"{path}: already migrated")
        return False
    if magic == LEGACY_V1_MAGIC:
        raise SystemExit(
            f"{path}: this is a v1 artifact, which was unsupported before the rename too. "
            f"It needs the v1->v2 migration, not this tool."
        )
    if magic != LEGACY_MAGIC:
        raise SystemExit(f"{path}: not a serving artifact (magic {magic!r})")

    target = path if keep_name else path.with_suffix(CURRENT_SUFFIX)
    if dry_run:
        print(f"{path}: would patch magic {LEGACY_MAGIC!r} -> {CURRENT_MAGIC!r}"
              + ("" if keep_name else f" and rename to {target.name}"))
        return True

    # Write the six identifier bytes only; the version and everything after it
    # stay exactly as they were.
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(CURRENT_MAGIC)
        handle.flush()
        os.fsync(handle.fileno())
    if target != path:
        path.rename(target)
    print(f"{target}: magic patched in place"
          + ("" if target == path else f" and renamed from {path.name}"))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rename_magic",
        description="Rewrite a serving artifact's magic in place after the project rename.",
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and touch nothing")
    parser.add_argument("--keep-name", action="store_true",
                        help="patch the magic but leave the filename alone")
    args = parser.parse_args(argv)

    changed = 0
    for path in args.artifacts:
        if not path.exists():
            raise SystemExit(f"{path}: no such file")
        changed += bool(migrate(path, dry_run=args.dry_run, keep_name=args.keep_name))
    print(f"{changed}/{len(args.artifacts)} artifact(s) "
          f"{'would change' if args.dry_run else 'changed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
