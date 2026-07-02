"""Generated train-allowed repository repair tasks.

These are small local repo bugs for the fast/medium Ultra curriculum. They use the
same patch-and-hidden-tests contract as the repo canary, but they are real GRPO
training candidates rather than smoke-test fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
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

DEFAULT_IMAGE_PREFIX = "fugu-ultra/generated-repo"
SOURCE_NAME = "generated_repo_tasks"
SOURCE_VERSION = "v1"


@dataclass(frozen=True)
class GeneratedRepoTask:
    task_id: str
    module_path: str
    initial_source: str
    instruction: str
    public_test: str
    hidden_test: str
    subdomain: str
    image_suffix: str


TASKS: tuple[GeneratedRepoTask, ...] = (
    GeneratedRepoTask(
        task_id="pathutils-safe-join",
        module_path="pathutils.py",
        initial_source="""\
import os


def safe_join(base: str, *parts: str) -> str:
    return os.path.normpath(os.path.join(base, *parts))
""",
        instruction="""Fix `safe_join` in `pathutils.py`.

Expected behavior:
- join `base` and path parts and return a normalized path
- reject absolute path parts by raising `ValueError`
- reject any path that escapes `base` after normalization by raising `ValueError`
- allow normalization that stays inside `base`

Edit source files only. Do not modify tests.
""",
        public_test="""\
from pathutils import safe_join


def test_nested_path():
    assert safe_join("/tmp/app", "data", "file.txt") == "/tmp/app/data/file.txt"
""",
        hidden_test="""\
import os
from pathutils import safe_join

assert safe_join("/tmp/app", "data/../x.txt") == "/tmp/app/x.txt"
for parts in [("../etc/passwd",), ("/etc/passwd",), ("data", "..", "..", "secret")]:
    try:
        safe_join("/tmp/app", *parts)
    except ValueError:
        pass
    else:
        raise AssertionError(f"unsafe path accepted: {parts}")
""",
        subdomain="path_security",
        image_suffix="pathutils-safe-join-v1",
    ),
    GeneratedRepoTask(
        task_id="statskit-percentile",
        module_path="statskit.py",
        initial_source="""\
def percentile(values, q):
    ordered = sorted(values)
    index = int((q / 100) * (len(ordered) - 1))
    return ordered[index]
""",
        instruction="""Fix `percentile` in `statskit.py`.

Expected behavior:
- accept `q` from 0 to 100 inclusive
- raise `ValueError` for an empty input or out-of-range `q`
- compute the percentile using linear interpolation between nearest ranks
- do not mutate the input sequence

Edit source files only. Do not modify tests.
""",
        public_test="""\
from statskit import percentile


def test_median_odd_count():
    assert percentile([3, 1, 2], 50) == 2
""",
        hidden_test="""\
from statskit import percentile

values = [4, 1, 3, 2]
assert percentile(values, 25) == 1.75
assert percentile(values, 50) == 2.5
assert percentile(values, 100) == 4
assert values == [4, 1, 3, 2]
for bad in [([], 50), ([1], -1), ([1], 101)]:
    try:
        percentile(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid percentile input accepted: {bad}")
""",
        subdomain="statistics",
        image_suffix="statskit-percentile-v1",
    ),
    GeneratedRepoTask(
        task_id="cachelite-ttl-expiry",
        module_path="cachelite.py",
        initial_source="""\
class TTLCache:
    def __init__(self, ttl):
        self.ttl = ttl
        self._items = {}

    def set(self, key, value, now):
        self._items[key] = (value, now)

    def get(self, key, now, default=None):
        if key not in self._items:
            return default
        value, inserted_at = self._items[key]
        if now > inserted_at + self.ttl:
            return default
        return value
""",
        instruction="""Fix `TTLCache.get` in `cachelite.py`.

Expected behavior:
- return `default` for missing keys
- expire an item when `now >= inserted_at + ttl`
- remove expired keys from the internal cache
- support `ttl=0` as immediate expiry

Edit source files only. Do not modify tests.
""",
        public_test="""\
from cachelite import TTLCache


def test_hit_before_expiry():
    cache = TTLCache(ttl=10)
    cache.set("a", 1, now=0)
    assert cache.get("a", now=9) == 1
""",
        hidden_test="""\
from cachelite import TTLCache

cache = TTLCache(ttl=10)
cache.set("a", 1, now=0)
assert cache.get("missing", now=5, default="x") == "x"
assert cache.get("a", now=10, default="expired") == "expired"
assert "a" not in cache._items
zero = TTLCache(ttl=0)
zero.set("z", 3, now=7)
assert zero.get("z", now=7, default=None) is None
assert "z" not in zero._items
""",
        subdomain="stateful_util",
        image_suffix="cachelite-ttl-expiry-v1",
    ),
    GeneratedRepoTask(
        task_id="csvlite-quoted-row",
        module_path="csvlite.py",
        initial_source="""\
def parse_csv_row(row: str) -> list[str]:
    return [part.strip() for part in row.split(",")]
""",
        instruction="""Fix `parse_csv_row` in `csvlite.py`.

Expected behavior:
- parse one CSV row into a list of strings
- support quoted fields containing commas
- support doubled quotes inside quoted fields
- preserve empty fields, including trailing empty fields
- trim whitespace around unquoted fields

Edit source files only. Do not modify tests.
""",
        public_test="""\
from csvlite import parse_csv_row


def test_simple_row():
    assert parse_csv_row("a, b,c") == ["a", "b", "c"]
""",
        hidden_test='''\
from csvlite import parse_csv_row

assert parse_csv_row('"a,b", c,') == ["a,b", "c", ""]
assert parse_csv_row('"say ""hi""",42') == ['say "hi"', "42"]
assert parse_csv_row(',,') == ["", "", ""]
assert parse_csv_row(' " spaced " ,x ') == [" spaced ", "x"]
''',
        subdomain="parsing",
        image_suffix="csvlite-quoted-row-v1",
    ),
    GeneratedRepoTask(
        task_id="semverlite-compare",
        module_path="semverlite.py",
        initial_source="""\
def compare_versions(left: str, right: str) -> int:
    if left == right:
        return 0
    return -1 if left < right else 1
""",
        instruction="""Fix `compare_versions` in `semverlite.py`.

Expected behavior:
- compare dot-separated numeric version components
- ignore trailing zero components
- return -1, 0, or 1
- raise `ValueError` for empty or non-numeric components

Edit source files only. Do not modify tests.
""",
        public_test="""\
from semverlite import compare_versions


def test_equal_versions():
    assert compare_versions("1.0", "1.0.0") == 0
""",
        hidden_test="""\
from semverlite import compare_versions

assert compare_versions("1.10", "1.2") == 1
assert compare_versions("2.0.0", "2") == 0
assert compare_versions("3.0.1", "3.0.2") == -1
for bad in [("", "1"), ("1..0", "1"), ("1.a", "1")]:
    try:
        compare_versions(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid version accepted: {bad}")
""",
        subdomain="comparison",
        image_suffix="semverlite-compare-v1",
    ),
    GeneratedRepoTask(
        task_id="textops-collapse-runs",
        module_path="textops.py",
        initial_source="""\
def collapse_runs(text: str, chars: str) -> str:
    for ch in chars:
        text = text.replace(ch + ch, ch)
    return text
""",
        instruction="""Fix `collapse_runs` in `textops.py`.

Expected behavior:
- collapse every consecutive run of characters from `chars` to a single character
- preserve the first character in each run
- apply repeatedly until no collapsible run remains
- leave runs of characters not listed in `chars` unchanged

Edit source files only. Do not modify tests.
""",
        public_test="""\
from textops import collapse_runs


def test_spaces():
    assert collapse_runs("a   b", " ") == "a b"
""",
        hidden_test="""\
from textops import collapse_runs

assert collapse_runs("a----b___c", "-_") == "a-b_c"
assert collapse_runs("111222", "12") == "12"
assert collapse_runs("xxxy", "z") == "xxxy"
assert collapse_runs("", " ") == ""
""",
        subdomain="string_processing",
        image_suffix="textops-collapse-runs-v1",
    ),
    GeneratedRepoTask(
        task_id="intervals-merge-touching",
        module_path="intervals.py",
        initial_source="""\
def merge_intervals(intervals):
    return sorted(intervals)
""",
        instruction="""Fix `merge_intervals` in `intervals.py`.

Expected behavior:
- accept an iterable of `(start, end)` pairs
- raise `ValueError` when `start > end`
- merge overlapping intervals and intervals that touch at a boundary
- return a new list sorted by start position
- do not mutate the input list

Edit source files only. Do not modify tests.
""",
        public_test="""\
from intervals import merge_intervals


def test_no_overlap_sorted():
    assert merge_intervals([(3, 4), (1, 2)]) == [(1, 2), (3, 4)]
""",
        hidden_test="""\
from intervals import merge_intervals

items = [(5, 6), (1, 3), (3, 4), (10, 10), (9, 9)]
assert merge_intervals(items) == [(1, 6), (9, 10)]
assert items == [(5, 6), (1, 3), (3, 4), (10, 10), (9, 9)]
try:
    merge_intervals([(2, 1)])
except ValueError:
    pass
else:
    raise AssertionError("invalid interval accepted")
""",
        subdomain="intervals",
        image_suffix="intervals-merge-touching-v1",
    ),
    GeneratedRepoTask(
        task_id="jsonpointer-escaped-get",
        module_path="jsonpointerlite.py",
        initial_source="""\
def get_pointer(document, pointer, default=None):
    if pointer == "":
        return document
    current = document
    for part in pointer.strip("/").split("/"):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current
""",
        instruction="""Fix `get_pointer` in `jsonpointerlite.py`.

Expected behavior:
- implement JSON Pointer lookup for dictionaries and lists
- support `~0` for `~` and `~1` for `/`
- return `default` instead of raising for missing keys, bad indexes, or malformed pointers
- return the whole document for an empty pointer

Edit source files only. Do not modify tests.
""",
        public_test="""\
from jsonpointerlite import get_pointer


def test_simple_dict_lookup():
    assert get_pointer({"a": {"b": 3}}, "/a/b") == 3
""",
        hidden_test="""\
from jsonpointerlite import get_pointer

doc = {"a/b": {"~key": [10, 20]}, "plain": 5}
assert get_pointer(doc, "") is doc
assert get_pointer(doc, "/a~1b/~0key/1") == 20
assert get_pointer(doc, "/a~1b/~0key/5", default="missing") == "missing"
assert get_pointer(doc, "not/a/pointer", default=None) is None
assert get_pointer(doc, "/plain/x", default="bad") == "bad"
""",
        subdomain="json",
        image_suffix="jsonpointer-escaped-get-v1",
    ),
    GeneratedRepoTask(
        task_id="ratecounter-window",
        module_path="ratecounter.py",
        initial_source="""\
class RateCounter:
    def __init__(self, window):
        self.window = window
        self.events = []

    def add(self, timestamp):
        self.events.append(timestamp)

    def count(self, now):
        return len([t for t in self.events if t >= now - self.window])
""",
        instruction="""Fix `RateCounter` in `ratecounter.py`.

Expected behavior:
- validate that `window` is non-negative
- `add(timestamp)` records an event timestamp
- `count(now)` counts events where `now - window < timestamp <= now`
- remove events older than the active window when counting
- ignore future events when counting

Edit source files only. Do not modify tests.
""",
        public_test="""\
from ratecounter import RateCounter


def test_counts_recent_event():
    counter = RateCounter(window=10)
    counter.add(5)
    assert counter.count(10) == 1
""",
        hidden_test="""\
from ratecounter import RateCounter

counter = RateCounter(window=10)
for ts in [0, 1, 5, 11, 30]:
    counter.add(ts)
assert counter.count(11) == 2
assert counter.events == [5, 11, 30]
assert counter.count(40) == 0
try:
    RateCounter(window=-1)
except ValueError:
    pass
else:
    raise AssertionError("negative window accepted")
""",
        subdomain="stateful_util",
        image_suffix="ratecounter-window-v1",
    ),
    GeneratedRepoTask(
        task_id="mdheadings-ignore-fences",
        module_path="mdheadings.py",
        initial_source="""\
def extract_headings(markdown: str) -> list[tuple[int, str]]:
    headings = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            headings.append((level, line[level:].strip()))
    return headings
""",
        instruction="""Fix `extract_headings` in `mdheadings.py`.

Expected behavior:
- return `(level, title)` tuples for ATX headings
- ignore headings inside fenced code blocks using triple backticks
- require a space after the heading marker
- strip optional closing `#` markers and surrounding whitespace
- support heading levels 1 through 6 only

Edit source files only. Do not modify tests.
""",
        public_test="""\
from mdheadings import extract_headings


def test_basic_heading():
    assert extract_headings("# Title") == [(1, "Title")]
""",
        hidden_test='''\
from mdheadings import extract_headings

doc = """# Title
```python
# not a heading
```
## Section ##
####### too many
#NoSpace
### Deep
"""
assert extract_headings(doc) == [(1, "Title"), (2, "Section"), (3, "Deep")]
''',
        subdomain="markdown",
        image_suffix="mdheadings-ignore-fences-v1",
    ),
    GeneratedRepoTask(
        task_id="topk-stable",
        module_path="ranker.py",
        initial_source="""\
def top_k(items, k, key=lambda x: x):
    return sorted(items, key=key)[-k:]
""",
        instruction="""Fix `top_k` in `ranker.py`.

Expected behavior:
- return the top `k` items by descending score
- preserve original order among equal scores
- return an empty list for `k=0`
- raise `ValueError` for negative `k`
- do not mutate the input sequence

Edit source files only. Do not modify tests.
""",
        public_test="""\
from ranker import top_k


def test_top_numbers():
    assert top_k([1, 3, 2], 2) == [3, 2]
""",
        hidden_test="""\
from ranker import top_k

items = [("a", 5), ("b", 7), ("c", 7), ("d", 1)]
assert top_k(items, 2, key=lambda item: item[1]) == [("b", 7), ("c", 7)]
assert top_k(items, 0, key=lambda item: item[1]) == []
assert items == [("a", 5), ("b", 7), ("c", 7), ("d", 1)]
try:
    top_k(items, -1, key=lambda item: item[1])
except ValueError:
    pass
else:
    raise AssertionError("negative k accepted")
""",
        subdomain="ranking",
        image_suffix="topk-stable-v1",
    ),
    GeneratedRepoTask(
        task_id="deepmerge-dicts",
        module_path="deepmerge.py",
        initial_source="""\
def merge_dicts(left, right):
    result = dict(left)
    result.update(right)
    return result
""",
        instruction="""Fix `merge_dicts` in `deepmerge.py`.

Expected behavior:
- recursively merge dictionaries
- values from `right` override values from `left`
- when both values are dictionaries, merge them instead of replacing
- return a new structure without mutating either input

Edit source files only. Do not modify tests.
""",
        public_test="""\
from deepmerge import merge_dicts


def test_shallow_override():
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}
""",
        hidden_test="""\
from deepmerge import merge_dicts

left = {"db": {"host": "localhost", "port": 5432}, "debug": False}
right = {"db": {"port": 5433}, "cache": {"enabled": True}}
merged = merge_dicts(left, right)
assert merged == {"db": {"host": "localhost", "port": 5433}, "debug": False, "cache": {"enabled": True}}
merged["db"]["host"] = "changed"
assert left["db"]["host"] == "localhost"
assert right["db"]["port"] == 5433
""",
        subdomain="data_structures",
        image_suffix="deepmerge-dicts-v1",
    ),
    GeneratedRepoTask(
        task_id="rangeset-parse-contains",
        module_path="rangeset.py",
        initial_source="""\
class RangeSet:
    def __init__(self, spec):
        self.values = {int(part) for part in spec.split(",") if part}

    def contains(self, value):
        return value in self.values
""",
        instruction="""Fix `RangeSet` in `rangeset.py`.

Expected behavior:
- parse comma-separated integers and inclusive ranges like `3-5`
- allow whitespace around tokens
- `contains(value)` returns whether an integer is included
- raise `ValueError` for empty tokens, non-numeric bounds, or reversed ranges

Edit source files only. Do not modify tests.
""",
        public_test="""\
from rangeset import RangeSet


def test_single_values():
    ranges = RangeSet("1,3")
    assert ranges.contains(1)
    assert not ranges.contains(2)
""",
        hidden_test="""\
from rangeset import RangeSet

ranges = RangeSet("1, 3-5, 10")
assert [ranges.contains(i) for i in range(1, 7)] == [True, False, True, True, True, False]
assert ranges.contains(10)
for bad in ["1,,2", "5-3", "x", "1-x"]:
    try:
        RangeSet(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid range accepted: {bad}")
""",
        subdomain="parsing",
        image_suffix="rangeset-parse-contains-v1",
    ),
    GeneratedRepoTask(
        task_id="duration-parse-units",
        module_path="duration.py",
        initial_source="""\
def parse_duration(text: str) -> int:
    return int(text)
""",
        instruction="""Fix `parse_duration` in `duration.py`.

Expected behavior:
- parse a duration string into seconds
- support `h`, `m`, and `s` units in any order, separated by spaces
- allow bare integers only when they are followed by a unit
- reject duplicate units, unknown units, negative numbers, and empty input

Edit source files only. Do not modify tests.
""",
        public_test="""\
from duration import parse_duration


def test_seconds():
    assert parse_duration("30s") == 30
""",
        hidden_test="""\
from duration import parse_duration

assert parse_duration("1h 30m 5s") == 5405
assert parse_duration("2m 1h") == 3720
for bad in ["", "10", "-1s", "1h 2h", "3d", "1 h"]:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid duration accepted: {bad}")
""",
        subdomain="parsing",
        image_suffix="duration-parse-units-v1",
    ),
    GeneratedRepoTask(
        task_id="slugger-unique-slugs",
        module_path="slugger.py",
        initial_source="""\
def unique_slugs(titles):
    return [title.lower().replace(" ", "-") for title in titles]
""",
        instruction="""Fix `unique_slugs` in `slugger.py`.

Expected behavior:
- normalize titles to lowercase URL slugs
- replace each run of non-alphanumeric characters with one hyphen
- strip leading and trailing hyphens
- use `item` for titles that normalize to empty
- append `-2`, `-3`, ... for repeated slugs while preserving order

Edit source files only. Do not modify tests.
""",
        public_test="""\
from slugger import unique_slugs


def test_simple_title():
    assert unique_slugs(["Hello World"]) == ["hello-world"]
""",
        hidden_test="""\
from slugger import unique_slugs

titles = ["Hello, World!", "Hello World", "!!!", "hello---world", "Python 3.11"]
assert unique_slugs(titles) == ["hello-world", "hello-world-2", "item", "hello-world-3", "python-3-11"]
""",
        subdomain="string_processing",
        image_suffix="slugger-unique-slugs-v1",
    ),
    GeneratedRepoTask(
        task_id="roman-parse-strict",
        module_path="roman.py",
        initial_source="""\
VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def parse_roman(text: str) -> int:
    return sum(VALUES[ch] for ch in text)
""",
        instruction="""Fix `parse_roman` in `roman.py`.

Expected behavior:
- parse standard Roman numerals from 1 to 3999
- support subtractive pairs IV, IX, XL, XC, CD, and CM
- reject empty strings, invalid characters, and non-canonical numerals
- raise `ValueError` for invalid input

Edit source files only. Do not modify tests.
""",
        public_test="""\
from roman import parse_roman


def test_simple():
    assert parse_roman("XII") == 12
""",
        hidden_test="""\
from roman import parse_roman

assert parse_roman("MCMXCIV") == 1994
assert parse_roman("XLII") == 42
for bad in ["", "IIII", "VX", "IC", "MMMM", "ABC"]:
    try:
        parse_roman(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid roman accepted: {bad}")
""",
        subdomain="parsing",
        image_suffix="roman-parse-strict-v1",
    ),
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _dockerfile() -> str:
    return "\n".join(
        [
            "FROM python:3.11-slim",
            "RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*",
            "WORKDIR /app",
            "COPY repo/ /app/",
            "RUN git init && git config user.email ultra@example.invalid && git config user.name Ultra && git add . && git commit -m initial",
            'CMD ["sleep", "9000"]',
            "",
        ]
    )


def _test_sh(hidden_test: str) -> str:
    return "\n".join(
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
            "import traceback",
            "",
            "try:",
            *["    " + line if line else "" for line in hidden_test.strip().splitlines()],
            "except Exception as exc:",
            "    open('/logs/verifier/reward.json', 'w').write(json.dumps({'reward': 0, 'error': str(exc), 'traceback': traceback.format_exc()}))",
            "else:",
            "    open('/logs/verifier/reward.json', 'w').write(json.dumps({'reward': 1}))",
            "PY",
            "",
        ]
    )


def image_tag_for(task: GeneratedRepoTask, image_prefix: str = DEFAULT_IMAGE_PREFIX) -> str:
    return f"{image_prefix}:{task.image_suffix}"


def write_generated_repo_context(root: Path, task: GeneratedRepoTask, image_prefix: str = DEFAULT_IMAGE_PREFIX) -> Path:
    task_dir = root / task.task_id
    repo_dir = task_dir / "repo"
    tests_dir = task_dir / "tests"
    _write(repo_dir / task.module_path, task.initial_source)
    _write(repo_dir / "README.md", f"# {task.task_id}\n\nGenerated Fugu-Ultra repo repair task.\n")
    _write(repo_dir / "test_public.py", task.public_test)
    _write(task_dir / "instruction.md", task.instruction)
    _write(task_dir / "Dockerfile", _dockerfile())
    _write(tests_dir / "test.sh", _test_sh(task.hidden_test))
    _write(
        task_dir / "task.json",
        json.dumps(
            {
                "task_id": task.task_id,
                "image_tag": image_tag_for(task, image_prefix),
                "instruction": task.instruction,
                "source": SOURCE_NAME,
                "version": SOURCE_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return task_dir


def build_image(task_dir: Path, image_tag: str) -> None:
    subprocess.run(["docker", "build", "-t", image_tag, str(task_dir)], check=True)


def _base_verifier_reward(image_tag: str, tests_dir: Path) -> dict[str, Any]:
    cid = ""
    try:
        proc = subprocess.run(
            ["docker", "run", "-d", "--rm", "-v", f"{tests_dir}:/tests:ro", image_tag, "sleep", "9000"],
            capture_output=True,
            text=True,
            check=True,
        )
        cid = proc.stdout.strip()
        subprocess.run(
            [
                "docker",
                "exec",
                cid,
                "bash",
                "-lc",
                "mkdir -p /logs/artifacts /logs/verifier && bash /tests/test.sh >/tmp/test.log 2>&1 || true",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        reward_text = subprocess.run(
            [
                "docker",
                "exec",
                cid,
                "bash",
                "-lc",
                "cat /logs/verifier/reward.json 2>/dev/null || true",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout.strip()
        try:
            payload = json.loads(reward_text) if reward_text else {}
        except json.JSONDecodeError:
            payload = {"raw": reward_text}
        reward = float(payload.get("reward", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
        return {"reward": reward, "payload": payload}
    finally:
        if cid:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)


def task_spec(task: GeneratedRepoTask, task_dir: Path, image_prefix: str = DEFAULT_IMAGE_PREFIX) -> TaskSpec:
    image_tag = image_tag_for(task, image_prefix)
    opencode_instance = {
        "image_name": image_tag,
        "instance_id": "",
        "problem_statement": task.instruction,
        "testbed": "/app",
        "activate": "",
        "task_id": task.task_id,
        "task_dir": str(task_dir),
        "tests_dir": str(task_dir / "tests"),
        "test_command": "bash /tests/test.sh",
        "grader": "generated_repo_tasks_v1",
    }
    group = f"{SOURCE_NAME}/{task.task_id}"
    return TaskSpec(
        task_id=f"{SOURCE_NAME}__{task.task_id}",
        capability="agentic_coding",
        source=SourceRef(
            name=SOURCE_NAME,
            version=SOURCE_VERSION,
            policy="train_allowed",
            url_or_ref=str(task_dir),
        ),
        input=TaskInput(
            messages=[{"role": "user", "content": task.instruction}],
            assets=[{"opencode_instance": opencode_instance}],
            repo=RepoRef(url=f"local://{SOURCE_NAME}/{task.task_id}", base_commit="generated-v1"),
        ),
        environment=EnvironmentSpec(
            harness="opencode",
            image=image_tag,
            cpu_limit=1,
            memory_mb=1024,
            disk_mb=1024,
            wall_time_seconds=900,
        ),
        grader=GraderSpec(
            type="deep_swe_hidden_tests",
            command=["bash", "/tests/test.sh"],
            success_threshold=1.0,
        ),
        splitting=SplittingSpec(
            group_id=group,
            split="grpo_train",
            contamination_group=group,
        ),
        metadata=TaskMetadata(
            domain="software_engineering",
            subdomain=task.subdomain,
            tags=["training_distribution", "generated_repo", "fast_repo"],
            requires_tools=True,
            estimated_worker_calls=1,
        ),
    )


def materialize_generated_repo_tasks(
    *,
    work_dir: Path,
    out_jsonl: Path,
    report_out: Path | None = None,
    image_prefix: str = DEFAULT_IMAGE_PREFIX,
    build: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    selected = list(TASKS[:limit] if limit is not None else TASKS)
    specs: list[TaskSpec] = []
    task_reports: list[dict[str, Any]] = []
    for task in selected:
        task_dir = write_generated_repo_context(work_dir, task, image_prefix=image_prefix)
        image_tag = image_tag_for(task, image_prefix)
        base_verifier = None
        if build:
            build_image(task_dir, image_tag)
            base_verifier = _base_verifier_reward(image_tag, task_dir / "tests")
            if base_verifier["reward"] >= 1.0:
                raise RuntimeError(f"base image unexpectedly passes hidden verifier: {task.task_id}")
        spec = task_spec(task, task_dir, image_prefix=image_prefix)
        specs.append(spec)
        task_reports.append(
            {
                "task_id": spec.task_id,
                "task_dir": str(task_dir),
                "image_tag": image_tag,
                "split": spec.splitting.split,
                "policy": spec.source.policy,
                "contamination_group": spec.splitting.contamination_group,
                "base_verifier": base_verifier,
            }
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for spec in specs:
            f.write(json.dumps(spec.model_dump(mode="json"), sort_keys=True) + "\n")

    report = {
        "version": "generated_repo_tasks_v1",
        "source": SOURCE_NAME,
        "task_count": len(specs),
        "work_dir": str(work_dir),
        "out_jsonl": str(out_jsonl),
        "image_prefix": image_prefix,
        "images_built": build,
        "base_validation_ready": all(
            t.get("base_verifier") is not None and t["base_verifier"]["reward"] == 0.0
            for t in task_reports
        )
        if build
        else None,
        "tasks": task_reports,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
