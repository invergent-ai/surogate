"""Generated fast repo canaries from the training distribution.

These tasks are intentionally tiny, train-allowed repository bugs. They are for
harness smoke tests and curriculum Tier 1/Tier 2 plumbing, not target evaluation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .schemas import (
    EnvironmentSpec,
    GraderSpec,
    RepoRef,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)

DEFAULT_IMAGE_TAG = "fugu-ultra/training-repo-canary:slugkit-v1"
DEFAULT_TASK_ID = "slugkit-normalize-title"

INSTRUCTION = """Fix `normalize_title` in `slugkit.py`.

Expected behavior:
- lowercase the title
- trim leading/trailing whitespace
- replace every run of non-alphanumeric characters with one hyphen
- strip leading/trailing hyphens
- return `untitled` if the normalized result is empty

Edit source files only. Do not modify tests.
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_slugkit_context(root: Path, image_tag: str = DEFAULT_IMAGE_TAG) -> Path:
    task_dir = root / DEFAULT_TASK_ID
    repo_dir = task_dir / "repo"
    tests_dir = task_dir / "tests"
    _write(
        repo_dir / "slugkit.py",
        "\n".join(
            [
                "def normalize_title(title: str) -> str:",
                "    return title.strip().lower().replace(' ', '-')",
                "",
            ]
        ),
    )
    _write(
        repo_dir / "README.md",
        "# slugkit\n\nTiny generated repo canary for Fugu-Ultra harness smoke tests.\n",
    )
    _write(
        repo_dir / "test_public.py",
        "\n".join(
            [
                "from slugkit import normalize_title",
                "",
                "",
                "def test_simple_spaces():",
                "    assert normalize_title('Hello World') == 'hello-world'",
                "",
            ]
        ),
    )
    _write(task_dir / "instruction.md", INSTRUCTION)
    _write(
        task_dir / "Dockerfile",
        "\n".join(
            [
                "FROM python:3.11-slim",
                "RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*",
                "WORKDIR /app",
                "COPY repo/ /app/",
                "RUN git init && git config user.email ultra@example.invalid && git config user.name Ultra && git add . && git commit -m initial",
                'CMD ["sleep", "9000"]',
                "",
            ]
        ),
    )
    _write(
        tests_dir / "test.sh",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                "mkdir -p /logs/verifier",
                "cd /app || { echo '{\"reward\": 0}' > /logs/verifier/reward.json; exit 1; }",
                "if [ -s /logs/artifacts/model.patch ]; then",
                "  if git apply --check /logs/artifacts/model.patch >/tmp/patch.log 2>&1; then",
                "    git apply /logs/artifacts/model.patch",
                "  elif git apply -R --check /logs/artifacts/model.patch >/tmp/patch-reverse.log 2>&1; then",
                "    true",
                "  else",
                "    echo '{\"reward\": 0, \"error\": \"patch did not apply\"}' > /logs/verifier/reward.json",
                "    exit 0",
                "  fi",
                "fi",
                "python - <<'PY'",
                "import json",
                "from slugkit import normalize_title",
                "",
                "cases = {",
                "    ' Hello,   World!! ': 'hello-world',",
                "    'Already--Slug': 'already-slug',",
                "    '___': 'untitled',",
                "    ' release/v2.0 ': 'release-v2-0',",
                "}",
                "failed = []",
                "for raw, expected in cases.items():",
                "    got = normalize_title(raw)",
                "    if got != expected:",
                "        failed.append({'input': raw, 'expected': expected, 'got': got})",
                "reward = 0 if failed else 1",
                "open('/logs/verifier/reward.json', 'w').write(json.dumps({'reward': reward, 'failed': failed}))",
                "PY",
                "",
            ]
        ),
    )
    _write(
        task_dir / "task.json",
        json.dumps(
            {
                "task_id": DEFAULT_TASK_ID,
                "image_tag": image_tag,
                "instruction": INSTRUCTION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return task_dir


def build_image(task_dir: Path, image_tag: str) -> None:
    subprocess.run(
        ["docker", "build", "-t", image_tag, str(task_dir)],
        check=True,
    )


def slugkit_spec(task_dir: Path, image_tag: str = DEFAULT_IMAGE_TAG) -> TaskSpec:
    opencode_instance = {
        "image_name": image_tag,
        "instance_id": "",
        "problem_statement": INSTRUCTION,
        "testbed": "/app",
        "activate": "",
        "task_id": DEFAULT_TASK_ID,
        "task_dir": str(task_dir),
        "tests_dir": str(task_dir / "tests"),
        "test_command": "bash /tests/test.sh",
        "grader": "training_repo_canary_v1",
    }
    return TaskSpec(
        task_id=f"training_repo_canary__{DEFAULT_TASK_ID}",
        capability="agentic_coding",
        source=SourceRef(
            name="training_repo_canary",
            version="v1",
            policy="train_allowed",
            url_or_ref=str(task_dir),
        ),
        input=TaskInput(
            messages=[{"role": "user", "content": INSTRUCTION}],
            assets=[{"opencode_instance": opencode_instance}],
            repo=RepoRef(url="local://training_repo_canary/slugkit", base_commit="generated-v1"),
        ),
        environment=EnvironmentSpec(
            harness="opencode",
            image=image_tag,
            cpu_limit=1,
            memory_mb=1024,
            disk_mb=1024,
            wall_time_seconds=600,
        ),
        grader=GraderSpec(
            type="deep_swe_hidden_tests",
            command=["bash", "/tests/test.sh"],
            success_threshold=1.0,
        ),
        splitting=SplittingSpec(
            group_id="training_repo_canary/slugkit",
            split="diagnostic",
            contamination_group="training_repo_canary/slugkit",
        ),
        metadata=TaskMetadata(
            domain="software_engineering",
            subdomain="python",
            tags=["training_distribution", "fast_repo_canary", "generated"],
            requires_tools=True,
            estimated_worker_calls=1,
        ),
    )


def materialize_training_repo_canaries(
    *,
    work_dir: Path,
    out_jsonl: Path,
    report_out: Path | None = None,
    image_tag: str = DEFAULT_IMAGE_TAG,
    build: bool = True,
) -> dict[str, Any]:
    task_dir = write_slugkit_context(work_dir, image_tag=image_tag)
    if build:
        build_image(task_dir, image_tag)
    spec = slugkit_spec(task_dir, image_tag=image_tag)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(json.dumps(spec.model_dump(mode="json"), sort_keys=True) + "\n")
    report = {
        "version": "training_repo_canary_v1",
        "source": "training_repo_canary",
        "task_id": spec.task_id,
        "task_dir": str(task_dir),
        "image_tag": image_tag,
        "image_built": build,
        "out_jsonl": str(out_jsonl),
        "split": spec.splitting.split,
        "policy": spec.source.policy,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
