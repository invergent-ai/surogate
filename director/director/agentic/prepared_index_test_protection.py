"""Fail-closed protection for benchmark-owned tests in prepared repositories."""

from __future__ import annotations

import base64
import json
import re
import zlib
from pathlib import PurePosixPath
from typing import Any, override

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


PAGER_SETUP = "export PAGER=cat GIT_PAGER=cat SYSTEMD_PAGER=cat LESS=-FRX\n"
PROTECTION_POLICY = (
    "prepared_index_test_blobs_restored_and_untracked_test_paths_removed_after_each_batch"
)
PREPARED_REPOSITORY_SETUP_TIMEOUT_S = 900
PROTECTED_TEST_RESTORE_SCRIPT_MAX_BYTES = 96 * 1024
ENVIRONMENT_SETUP_RE = re.compile(
    r"^## Environment Setup[^\n]*\n.*?"
    r"```(?:bash|sh)\s*\n(?P<script>.*?)\n```[ \t]*",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
TASK_BODY_DELIMITER_RE = re.compile(r"^---[ \t]*$", re.MULTILINE)


def repository_discovery_python(roots: tuple[str, ...]) -> str:
    """Return container-side Python that discovers one unambiguous worktree.

    Harbor tasks commonly clone either directly into ``/testbed`` or into a
    child such as ``/testbed/repo``.  All protection and recovery boundaries
    must resolve the same concrete worktree instead of silently treating the
    parent directory as the repository.
    """
    encoded_roots = json.dumps(list(roots), ensure_ascii=True)
    return f"""
roots = {encoded_roots}

def discover_repository():
    repositories = set()
    for raw_root in roots:
        root = pathlib.Path(raw_root)
        if not root.is_dir():
            continue
        pending = [root]
        while pending:
            candidate = pending.pop()
            git_marker = candidate / '.git'
            if git_marker.is_dir() or git_marker.is_file():
                probe = run('git', '-C', str(candidate), 'rev-parse', '--show-toplevel')
                if probe.returncode == 0:
                    resolved = pathlib.Path(
                        probe.stdout.decode(errors='replace').strip()
                    ).resolve()
                    try:
                        resolved.relative_to(root.resolve())
                    except ValueError:
                        raise SystemExit(
                            'discovered Git worktree escapes workspace root: ' + str(resolved)
                        )
                    repositories.add(str(resolved))
                    # A repository is the ownership boundary. Do not discover
                    # nested submodules or tool caches as independent roots.
                    continue
            try:
                children = sorted(candidate.iterdir(), key=lambda path: path.name)
            except OSError:
                continue
            pending.extend(
                child
                for child in reversed(children)
                if child.is_dir()
                and not child.is_symlink()
                and child.name not in {{'.git', '.venv', '.tox', 'node_modules'}}
            )
    if len(repositories) > 1:
        raise SystemExit(
            'multiple Git worktrees found beneath workspace roots: '
            + ', '.join(sorted(repositories))
        )
    return next(iter(repositories), None)
"""


def split_environment_setup(instruction: str) -> tuple[str | None, str]:
    """Extract only the leading benchmark-declared shell setup block."""
    match = ENVIRONMENT_SETUP_RE.search(instruction)
    if match is None:
        return None, instruction
    prefix = instruction[: match.start()]
    if TASK_BODY_DELIMITER_RE.search(prefix):
        # Setup-like prose in the task body is user content, not a trusted
        # benchmark preparation directive.
        return None, instruction
    script = match.group("script").strip()
    if not script:
        raise ValueError("benchmark environment setup block is empty")
    suffix = instruction[match.end() :].lstrip()
    if suffix.startswith("---"):
        suffix = suffix[3:].lstrip()
    task_instruction = prefix.rstrip()
    if task_instruction and suffix:
        task_instruction += "\n\n" + suffix
    else:
        task_instruction += suffix
    if not task_instruction:
        raise ValueError("benchmark instruction contains setup but no task")
    return script, task_instruction


def is_benchmark_owned_test_path(path: str) -> bool:
    """Return whether a tracked repository path is benchmark-owned test input."""
    pure = PurePosixPath(path)
    parts = tuple(part.lower() for part in pure.parts)
    name = pure.name.lower()
    return (
        any(
            part in {"test", "tests", "testing", "spec", "specs"}
            for part in parts[:-1]
        )
        or name.startswith("test_")
        or name.endswith(("_test.py", "_test.go", "_test.rs", ".spec.js", ".test.js"))
    )


def protected_test_snapshot_script(
    roots: tuple[str, ...] = ("/testbed", "/app")
) -> str:
    """Snapshot protected test blobs from the prepared repository index."""
    discovery = repository_discovery_python(roots)
    return f"""python3 - <<'PY'
import json
import pathlib
import subprocess

def run(*args):
    try:
        return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 127, b'', b'git binary missing')

def protected(path):
    pure = pathlib.PurePosixPath(path)
    parts = tuple(part.lower() for part in pure.parts)
    name = pure.name.lower()
    return (
        any(part in {{'test', 'tests', 'testing', 'spec', 'specs'}} for part in parts[:-1])
        or name.startswith('test_')
        or name.endswith(('_test.py', '_test.go', '_test.rs', '.spec.js', '.test.js'))
    )

{discovery}
repo = discover_repository()
if repo is None:
    print(json.dumps({{'repo': None, 'entries': []}}))
    raise SystemExit(0)

listed = run('git', '-C', repo, 'ls-files', '-s', '-z')
if listed.returncode != 0:
    raise SystemExit(listed.stderr.decode(errors='replace'))
entries = []
for raw in listed.stdout.decode(errors='surrogateescape').split('\\0'):
    if not raw:
        continue
    metadata, separator, path = raw.partition('\\t')
    fields = metadata.split()
    if not separator or len(fields) != 3 or fields[2] != '0':
        raise SystemExit('unexpected git index entry: ' + raw)
    if protected(path):
        entries.append({{'path': path, 'mode': fields[0], 'oid': fields[1]}})
print(json.dumps({{'repo': repo, 'entries': entries}}, sort_keys=True))
PY"""


def protected_test_restore_script(
    baseline: list[dict[str, str]],
    roots: tuple[str, ...] = ("/testbed", "/app"),
) -> str:
    """Restore protected tests to their prepared, immutable index blobs."""
    discovery = repository_discovery_python(roots)
    encoded_baseline = base64.b64encode(
        zlib.compress(
            json.dumps(
                baseline, ensure_ascii=True, separators=(",", ":")
            ).encode("utf-8"),
            level=9,
        )
    ).decode("ascii")
    return f"""python3 - <<'PY'
import base64
import json
import pathlib
import subprocess
import zlib

baseline = json.loads(zlib.decompress(base64.b64decode('{encoded_baseline}')))

def run(*args):
    try:
        return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 127, b'', b'git binary missing')

{discovery}
repo = discover_repository()
if repo is None:
    print(json.dumps({{'repo': None, 'restored': []}}))
    raise SystemExit(0)

listed = run('git', '-C', repo, 'ls-files', '-s', '-z')
if listed.returncode != 0:
    raise SystemExit(listed.stderr.decode(errors='replace'))
current_index = {{}}
for raw in listed.stdout.decode(errors='surrogateescape').split('\\0'):
    if not raw:
        continue
    metadata, separator, path = raw.partition('\\t')
    fields = metadata.split()
    if separator and len(fields) == 3 and fields[2] == '0':
        current_index[path] = (fields[0], fields[1])
unstaged = run('git', '-C', repo, 'diff-files', '--name-only', '-z')
if unstaged.returncode != 0:
    raise SystemExit(unstaged.stderr.decode(errors='replace'))
worktree_changed = set(
    unstaged.stdout.decode(errors='surrogateescape').split('\\0')
)

changed = []
for entry in baseline:
    path = entry['path']
    mode = entry['mode']
    oid = entry['oid']
    if current_index.get(path) != (mode, oid) or path in worktree_changed:
        changed.append(path)
        object_check = run('git', '-C', repo, 'cat-file', '-e', oid + '^{{blob}}')
        if object_check.returncode != 0:
            raise SystemExit('missing protected blob ' + oid + ' for ' + path)
        reset = run(
            'git', '-C', repo, 'update-index', '--add', '--cacheinfo', mode, oid, path
        )
        if reset.returncode != 0:
            raise SystemExit(reset.stderr.decode(errors='replace'))
        restored = run('git', '-C', repo, 'checkout-index', '--force', '--', path)
        if restored.returncode != 0:
            raise SystemExit(restored.stderr.decode(errors='replace'))

def protected(path):
    pure = pathlib.PurePosixPath(path)
    parts = tuple(part.lower() for part in pure.parts)
    name = pure.name.lower()
    return (
        any(part in {{'test', 'tests', 'testing', 'spec', 'specs'}} for part in parts[:-1])
        or name.startswith('test_')
        or name.endswith(('_test.py', '_test.go', '_test.rs', '.spec.js', '.test.js'))
    )

untracked = run('git', '-C', repo, 'ls-files', '--others', '--exclude-standard', '-z')
if untracked.returncode != 0:
    raise SystemExit(untracked.stderr.decode(errors='replace'))
removed = []
for path in untracked.stdout.decode(errors='surrogateescape').split('\\0'):
    if not path or not protected(path):
        continue
    target = pathlib.Path(repo) / path
    if target.is_symlink() or target.is_file():
        target.unlink()
        removed.append(path)
print(json.dumps({{'repo': repo, 'restored': changed, 'removed': removed}}, sort_keys=True))
PY"""


def protected_test_restore_scripts(
    baseline: list[dict[str, str]],
    roots: tuple[str, ...] = ("/testbed", "/app"),
) -> list[str]:
    """Build restore commands bounded below Docker's per-argument limit."""
    if not baseline:
        return [protected_test_restore_script([], roots)]
    scripts: list[str] = []
    chunk: list[dict[str, str]] = []
    for entry in baseline:
        candidate = [*chunk, entry]
        candidate_script = protected_test_restore_script(candidate, roots)
        if (
            chunk
            and len(candidate_script.encode("utf-8"))
            > PROTECTED_TEST_RESTORE_SCRIPT_MAX_BYTES
        ):
            scripts.append(protected_test_restore_script(chunk, roots))
            candidate = [entry]
            candidate_script = protected_test_restore_script(candidate, roots)
        if len(candidate_script.encode("utf-8")) > PROTECTED_TEST_RESTORE_SCRIPT_MAX_BYTES:
            raise ValueError("one protected-test entry exceeds restore command limit")
        chunk = candidate
    if chunk:
        scripts.append(protected_test_restore_script(chunk, roots))
    return scripts


def prepared_git_history_sanitization_script(
    roots: tuple[str, ...] = ("/testbed", "/app")
) -> str:
    """Replace benchmark Git history with one tracked challenge baseline."""
    discovery = repository_discovery_python(roots)
    return f"""python3 - <<'PY'
import json
import pathlib
import shutil
import subprocess

def run(*args, input_bytes=None):
    return subprocess.run(
        args,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

{discovery}
repo = discover_repository()
if repo is None:
    raise SystemExit('prepared Git-history sanitization found no worktree')

listed = run('git', '-C', repo, 'ls-files', '-z')
if listed.returncode != 0 or not listed.stdout:
    raise SystemExit(listed.stderr.decode(errors='replace') or 'prepared worktree has no tracked files')
tracked_count = len([path for path in listed.stdout.split(b'\\0') if path])
git_dir = pathlib.Path(repo) / '.git'
if not git_dir.exists():
    raise SystemExit('prepared worktree has no local .git directory')
shutil.rmtree(git_dir)

for command in (
    ('git', '-C', repo, 'init', '-q'),
    ('git', '-C', repo, 'config', 'user.name', 'Fugu Benchmark Harness'),
    ('git', '-C', repo, 'config', 'user.email', 'benchmark@localhost'),
):
    result = run(*command)
    if result.returncode != 0:
        raise SystemExit(result.stderr.decode(errors='replace'))
added = run(
    'git', '-C', repo, 'add', '--force', '--pathspec-from-file=-', '--pathspec-file-nul',
    input_bytes=listed.stdout,
)
if added.returncode != 0:
    raise SystemExit(added.stderr.decode(errors='replace'))
committed = run('git', '-C', repo, 'commit', '-qm', 'Prepared challenge baseline')
if committed.returncode != 0:
    raise SystemExit(committed.stderr.decode(errors='replace'))

head = run('git', '-C', repo, 'rev-parse', 'HEAD')
commit_count = run('git', '-C', repo, 'rev-list', '--all', '--count')
remotes = run('git', '-C', repo, 'remote')
if head.returncode != 0 or commit_count.returncode != 0 or remotes.returncode != 0:
    raise SystemExit('failed to attest sanitized Git repository')
payload = {{
    'repo': repo,
    'head': head.stdout.decode().strip(),
    'commit_count': int(commit_count.stdout.decode().strip()),
    'remotes': [line for line in remotes.stdout.decode().splitlines() if line],
    'tracked_entries': tracked_count,
}}
if payload['commit_count'] != 1 or payload['remotes']:
    raise SystemExit('prepared Git-history sanitization invariant failed')
print(json.dumps(payload, sort_keys=True))
PY"""


class PreparedIndexTestProtectionMixin:
    """Restore prepared-index tests after every terminal command batch."""

    _sanitize_prepared_git_history = False

    def _initialize_protected_test_protection(self) -> None:
        self._protected_test_restores: list[dict[str, Any]] = []
        self._untracked_test_removals: list[dict[str, Any]] = []
        self._protected_test_snapshot: list[dict[str, str]] = []
        self._protected_test_environment: BaseEnvironment | None = None
        self._protected_test_repo: str | None = None
        self._prepared_repository_setup_executed = False
        self._prepared_git_history_sanitization: dict[str, Any] | None = None

    async def _snapshot_protected_tests(self) -> list[dict[str, str]]:
        if self._protected_test_environment is None:
            raise RuntimeError("protected-test environment is not initialized")
        result = await self._protected_test_environment.exec(
            protected_test_snapshot_script(), timeout_sec=30
        )
        if result.return_code != 0:
            raise RuntimeError(
                "tracked-test snapshot failed closed: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        payload = json.loads(result.stdout or "{}")
        repo = payload.get("repo")
        if repo is not None and not isinstance(repo, str):
            raise RuntimeError("tracked-test snapshot returned an invalid repository")
        entries = payload.get("entries") or []
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict)
            or set(entry) != {"path", "mode", "oid"}
            or not all(isinstance(value, str) for value in entry.values())
            for entry in entries
        ):
            raise RuntimeError("tracked-test snapshot returned invalid evidence")
        self._protected_test_repo = repo
        self._protected_test_snapshot = entries
        return entries

    async def _run_prepared_repository_setup(self, script: str) -> None:
        if self._protected_test_environment is None:
            raise RuntimeError("protected-test environment is not initialized")
        result = await self._protected_test_environment.exec(
            script,
            cwd="/",
            timeout_sec=PREPARED_REPOSITORY_SETUP_TIMEOUT_S,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(
                "benchmark repository setup failed before paid work: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        self._prepared_repository_setup_executed = True

    async def _sanitize_prepared_repository_history(self) -> None:
        if not self._sanitize_prepared_git_history:
            return
        if self._protected_test_environment is None:
            raise RuntimeError("protected-test environment is not initialized")
        result = await self._protected_test_environment.exec(
            prepared_git_history_sanitization_script(), timeout_sec=120
        )
        if result.return_code != 0:
            raise RuntimeError(
                "prepared Git-history sanitization failed closed: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        payload = json.loads(result.stdout or "{}")
        if (
            not isinstance(payload, dict)
            or payload.get("commit_count") != 1
            or payload.get("remotes") != []
            or not isinstance(payload.get("head"), str)
            or not isinstance(payload.get("tracked_entries"), int)
        ):
            raise RuntimeError(
                "prepared Git-history sanitization returned invalid evidence"
            )
        self._prepared_git_history_sanitization = payload

    async def _prepare_protected_repository(self, instruction: str) -> str:
        setup_script, task_instruction = split_environment_setup(instruction)
        await self._snapshot_protected_tests()
        if self._protected_test_repo is None and setup_script is not None:
            await self._run_prepared_repository_setup(setup_script)
            await self._snapshot_protected_tests()
            if self._protected_test_repo is None:
                raise RuntimeError(
                    "benchmark repository setup completed without a Git worktree"
                )
        if self._protected_test_repo is not None:
            await self._sanitize_prepared_repository_history()
            if self._sanitize_prepared_git_history:
                await self._snapshot_protected_tests()
        return task_instruction

    async def _restore_protected_tests(self) -> list[str]:
        if self._protected_test_environment is None:
            raise RuntimeError("protected-test environment is not initialized")
        if self._protected_test_repo is None:
            # Pin protection to the prepared baseline. A task may legitimately
            # create a Git repository after starting in an empty workspace.
            return []
        restored: list[str] = []
        removed: list[str] = []
        for script in protected_test_restore_scripts(self._protected_test_snapshot):
            result = await self._protected_test_environment.exec(
                script,
                timeout_sec=30,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    "tracked-test protection failed closed: "
                    f"{(result.stderr or result.stdout or '').strip()}"
                )
            payload = json.loads(result.stdout or "{}")
            chunk_restored = payload.get("restored") or []
            chunk_removed = payload.get("removed") or []
            if (
                payload.get("repo") != self._protected_test_repo
                or not isinstance(chunk_restored, list)
                or not all(isinstance(path, str) for path in chunk_restored)
                or not isinstance(chunk_removed, list)
                or not all(isinstance(path, str) for path in chunk_removed)
            ):
                raise RuntimeError("tracked-test protection returned invalid evidence")
            restored.extend(chunk_restored)
            removed.extend(path for path in chunk_removed if path not in removed)
        fugu_llm = getattr(self, "_fugu_llm", None)
        after_paid_call = getattr(fugu_llm, "paid_worker_call_attempts", 0)
        if restored:
            # Tracked benchmark-owned tests were modified by the run; this
            # ledger keeps its historical tampering semantics for admission.
            self._protected_test_restores.append(
                {
                    "repo": self._protected_test_repo,
                    "paths": restored,
                    "after_paid_call": after_paid_call,
                }
            )
        if removed:
            # Benign harness cleanup of agent-created files at verifier-owned
            # paths; recorded separately so it never reads as tampering.
            self._untracked_test_removals.append(
                {
                    "repo": self._protected_test_repo,
                    "paths": removed,
                    "after_paid_call": after_paid_call,
                }
            )
        return restored + removed

    @override
    async def _execute_commands(
        self, commands: list[Any], session: Any
    ) -> tuple[bool, str]:
        timeout_occurred, terminal_output = await super()._execute_commands(
            commands, session
        )
        restores_before = len(self._protected_test_restores)
        removals_before = len(self._untracked_test_removals)
        await self._restore_protected_tests()
        if len(self._protected_test_restores) > restores_before:
            terminal_output += (
                "\nHARNESS SAFETY: benchmark-owned test files were restored from "
                "the prepared baseline: "
                + ", ".join(self._protected_test_restores[-1]["paths"])
                + ". Inspect and run them, but do not modify them.\n"
            )
        if len(self._untracked_test_removals) > removals_before:
            terminal_output += (
                "\nHARNESS SAFETY: new files are not allowed inside the "
                "benchmark-owned test namespace and were removed: "
                + ", ".join(self._untracked_test_removals[-1]["paths"])
                + ". Keep any files you create outside protected test paths.\n"
            )
        return timeout_occurred, terminal_output

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._session is None:
            raise RuntimeError("terminal session is not initialized")
        self._protected_test_environment = environment
        self._protected_test_restores.clear()
        self._untracked_test_removals.clear()
        self._protected_test_snapshot.clear()
        self._protected_test_repo = None
        self._prepared_repository_setup_executed = False
        self._prepared_git_history_sanitization = None
        await self._session.send_keys(PAGER_SETUP, block=True, min_timeout_sec=0.1)
        task_instruction = await self._prepare_protected_repository(instruction)
        await super().run(task_instruction, environment, context)

    def _protected_test_metadata(self) -> dict[str, Any]:
        return {
            "noninteractive_pager_environment": True,
            "protected_test_restore_policy": PROTECTION_POLICY,
            "protected_test_repo": self._protected_test_repo,
            "protected_test_snapshot_entries": len(self._protected_test_snapshot),
            "protected_test_restores": self._protected_test_restores,
            "untracked_test_removals": self._untracked_test_removals,
            "prepared_repository_setup_executed": (
                self._prepared_repository_setup_executed
            ),
            "prepared_git_history_sanitization": (
                self._prepared_git_history_sanitization
            ),
        }
