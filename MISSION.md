# Mission: Fugu-Ultra Build And Evaluation

Objective: create a highly performant Fugu-Ultra model that outperforms any individual model or individual model+scaffold worker on held-out workflow outcomes.

Method:
- Use a quality-first frontier/specialist worker pool.
- Train a Conductor on executed multi-step workflows.
- Compare against individual frontier models, individual scaffolded workers, best fixed workflows, and best-of-N single-scaffold baselines.
- Select workers by paired held-out workflow contribution, not direct accuracy alone.

## Status

Last updated: 2026-06-27 03:56 UTC.

Core pool:
- `opus`
- `gemini`
- `gpt`
- `kimi-code`
- `mimo`
- `glm`
- `flash`

Optional challengers:
- `minimax`
- `deepseek-pro`

Current readiness:
- 321 preregistered jobs / 603 worker calls.
- 96 direct-reasoning jobs are ready.
- 36 OpenCode + local Deep SWE jobs are bridge-ready but not live-smoked.
- 9 saved live SWE-smith coding jobs are missing original payloads.
- 120 Codex/Claude coding jobs need concrete adapters.
- 60 tau/tool-dialog jobs need a concrete harness.
- 24 TaskTrove inferredbugs Harbor TaskSpecs are materialized for canary.
- Harbor CLI is installed; no-model Docker/verifier canary completed.

Current blockers:
- Run one OpenCode + Deep SWE Docker canary before any repo-coding rollout.
- Recover or replace the three saved live SWE-smith tasks without payloads.
- Implement concrete `ClaudeCodeHarness`, `CodexHarness`, and `ToolDialogHarness`.
- Configure Harbor model provider env and run one model-backed TaskTrove canary before tournament inclusion.

Operating constraints:
- Do not write secrets to repo files, logs, artifacts, command arguments, or mission notes.
- Do not treat direct-only jobs as proof of Ultra performance.

## Milestones

- [x] Read Ultra docs and inspect existing direct, tau, frontier, and coding evidence.
- [x] Correct pool principle to quality-first frontier triad plus useful specialist workers.
- [x] Define scaffold-aware worker identity: `model + scaffold + settings`.
- [x] Convert saved OpenCode rollouts into canonical traces.
- [x] Preregister scaffold-aware role tournament.
- [x] Generate concrete tournament task/job manifest.
- [x] Correct executor to route each step by worker scaffold.
- [x] Materialize local Deep SWE repo-coding tasks for OpenCode.
- [x] Add OpenCode Deep SWE grading bridge.
- [x] Study TaskTrove/Harbor relevance for diversity expansion.
- [x] Implement Harbor `terminal_sandbox` adapter.
- [x] Select verifier-backed TaskTrove subsets.
- [x] Live-smoke one no-model Harbor TaskTrove verifier run.
- [ ] Live-smoke one OpenCode + Deep SWE task.
- [ ] Resolve three payload-missing live SWE-smith tasks.
- [ ] Implement Claude Code harness.
- [ ] Implement Codex harness.
- [ ] Implement tau/tool-dialog harness.
- [ ] Run model-backed Harbor TaskTrove baseline/canary.
- [ ] Add Harbor `terminal_sandbox` TaskSpecs/traces to the tournament.
- [ ] Run scaffold-aware tournament.
- [ ] Analyze paired outcomes and leave-one-out contribution.
- [ ] Select final training pool.
- [ ] Build GRPO pilot data.
- [ ] Train Fugu-Ultra Conductor/model.
- [ ] Evaluate against individual models, scaffolded workers, fixed workflows, and best-of-N baselines.

## Handover Log

### 2026-06-26 - Evidence Review

Done:
- Read Ultra docs and existing evidence files.
- Confirmed Ultra must be evaluated as workflow orchestration, not a router or direct-only benchmark.

Result:
- Flash has strong direct/open QA signal.
- MiMo has strong tau/tool-dialog signal.
- Kimi and MiMo have strongest observed OpenCode coding signal.
- Opus/Gemini coding failures in the tiny shard were real patch failures, not harness/tool-call failures.

### 2026-06-27 - Pool Principle Corrected

Done:
- Switched from “cheap coding workers” to quality-first Fugu-Ultra pool.

Result:
- Main pool is `opus`, `gemini`, `gpt`, `kimi-code`, `mimo`, `glm`, `flash`.
- `minimax` and `deepseek-pro` remain challengers/ablations.

### 2026-06-27 - Scaffold-Aware Design

Done:
- Changed worker identity from model-only to model+scaffold.
- Added canonical trace path for saved OpenCode rollouts.

Result:
- Saved OpenCode traces are available at `director/manifests/fugu_clean_v1/agent_traces/opencode_direct3/traces.jsonl`.
- Those traces are outcome-level only; they are not sufficient for state-level training.

### 2026-06-27 - Tournament Preregistered

Done:
- Preregistered scaffold-aware workers, role workflows, baselines, and decision rule.
- Selected concrete tasks and jobs.

Result:
- Plan: `director/manifests/fugu_clean_v1/scaffold_tournament_plan.json`
- Manifest: `director/manifests/fugu_clean_v1/scaffold_tournament_manifest.json`
- Jobs: `director/manifests/fugu_clean_v1/scaffold_tournament_jobs.jsonl`
- Mix: 15 repo-coding, 10 tau/tool-dialog, 12 direct reasoning.

### 2026-06-27 - Readiness Established

Done:
- Generated readiness report for the tournament jobs.

Result:
- Readiness report: `director/manifests/fugu_clean_v1/scaffold_tournament_readiness.json`
- Direct jobs are ready.
- OpenCode local Deep SWE jobs need live smoke.
- Codex, Claude Code, and tool-dialog execution are still pending.

### 2026-06-27 - Executor Routing Corrected

Done:
- Corrected Ultra executor so each step routes by worker scaffold.

Result:
- A single workflow can now mix `codex`, `claude_code`, `opencode`, `direct_qa`, and `tool_dialog`.
- This fixes the design issue where repo-coding workflows were being treated as one task-level harness.

### 2026-06-27 - Repo-Coding Payloads Prepared

Done:
- Materialized local Deep SWE repo tasks into OpenCode-ready task specs.
- Added bridge for OpenCode to write `model.patch`, run `/tests/test.sh`, and parse reward output.

Result:
- TaskSpecs: `director/manifests/fugu_clean_v1/scaffold_repo_taskspecs.jsonl`
- Report: `director/manifests/fugu_clean_v1/scaffold_repo_taskspec_report.json`
- 12 local Deep SWE tasks are materialized.
- 3 saved live SWE-smith tasks remain unresolved because their original payloads are missing.

Next:
- Run one OpenCode + Deep SWE canary.

### 2026-06-27 - Mission Log Cleanup

Done:
- Removed verbose historical notes, test counts, and code-change inventories.

Result:
- This file is now a handover document: status, milestones, concise action/result notes, and reproducible commands.

### 2026-06-27 - TaskTrove/Harbor Study

Done:
- Checked current HF TaskTrove page and local OpenThoughts-Agent Harbor tooling.
- Compared Harbor task bundles against Ultra `terminal_sandbox` needs.

Result:
- Relevant and useful as a diversity expansion.
- Use Harbor for verified terminal/sandbox tasks and TaskTrove traces, not as a replacement for OpenCode/Codex/Claude repo-coding harnesses.
- Prefer verifier-backed subsets for evaluation/GRPO; no-verifier subsets are trace/source material only.
- Local `harbor` CLI is not installed yet.

Next:
- Continue with Harbor subset selection and canary.

### 2026-06-27 - Harbor Lane Prepared

Done:
- Added Ultra materialization for Harbor task bundles.
- Added fail-closed `terminal_sandbox` execution through Harbor.
- Selected TaskTrove inferredbugs as the first canary shard and staged local TaskSpecs.

Result:
- Subset plan: `director/manifests/fugu_clean_v1/tasktrove_harbor/subset_selection.json`
- Canary TaskSpecs: `director/manifests/fugu_clean_v1/tasktrove_harbor/inferredbugs_canary_taskspecs.jsonl`
- Harbor execution path was ready but still needed a live canary at this point.

Next:
- Run one model-backed Harbor canary with a scaffold-aware baseline.

### 2026-06-27 - Harbor Canary Passed

Done:
- Installed Harbor CLI with `uv tool`.
- Ran one TaskTrove inferredbugs task through Harbor Docker using `nop`.

Result:
- Harbor loaded the task, built/ran Docker, executed the verifier, and produced a result with no exceptions.
- Reward was 0.0, expected for `nop`.
- Result: `director/manifests/fugu_clean_v1/tasktrove_harbor/harbor_jobs/fugu_tasktrove_nop_canary/result.json`

Next:
- Configure Harbor model provider env, then run the same canary with a real agent and include Harbor solo as a baseline.

## Repro Commands

Regenerate OpenCode trace conversion:

```bash
cd /home/densemax/work/flavius/surogate/ultra
.venv/bin/python -m ultra.traces.opencode_rollouts \
  /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/agentic_coding_frontier_direct3.jsonl \
  --out-dir /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/agent_traces/opencode_direct3
```

Regenerate scaffold tournament plan:

```bash
cd /home/densemax/work/flavius/surogate/ultra
.venv/bin/python -m ultra.cli scaffold-tournament-plan \
  --coding-tasks 15 \
  --tool-dialog-tasks 10 \
  --direct-tasks 12 \
  --out /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/scaffold_tournament_plan.json
```

Regenerate concrete tournament manifest/jobs:

```bash
cd /home/densemax/work/flavius/surogate/ultra
.venv/bin/python -m ultra.cli scaffold-tournament-manifest \
  --manifest-dir /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1 \
  --coding-tasks 15 \
  --tool-dialog-tasks 10 \
  --direct-tasks 12 \
  --seed 0 \
  --out /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/scaffold_tournament_manifest.json \
  --jobs-out /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/scaffold_tournament_jobs.jsonl
```

Regenerate readiness report:

```bash
cd /home/densemax/work/flavius/surogate/ultra
.venv/bin/python -m ultra.cli scaffold-tournament-readiness \
  /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/scaffold_tournament_manifest.json \
  --out /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/scaffold_tournament_readiness.json
```

Regenerate repo TaskSpecs:

```bash
cd /home/densemax/work/flavius/surogate/ultra
.venv/bin/python -m ultra.cli scaffold-materialize-repo \
  /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/scaffold_tournament_manifest.json \
  --out-jsonl /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/scaffold_repo_taskspecs.jsonl \
  --report-out /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/scaffold_repo_taskspec_report.json
```

Run focused verification:

```bash
cd /home/densemax/work/flavius/surogate/ultra
.venv/bin/python -m pytest \
  tests/test_schemas.py \
  tests/test_traces.py \
  tests/test_opencode_trace_conversion.py \
  tests/test_scaffold_tournament.py \
  tests/test_scaffold_materialize.py \
  tests/test_executor.py \
  tests/test_adapters.py \
  tests/test_pool_selection.py \
  tests/test_pool_tournament.py \
  tests/test_opencode_harness.py \
  tests/test_harbor_source.py \
  tests/test_harbor_harness.py \
  tests/test_stepzero_cli.py
```

Inspect local TaskTrove v2 catalog:

```bash
cd /home/densemax/work/flavius/surogate/study/OpenThoughts-Agent
python data/tasktrove/tasktrove_v2_datasets.py
```

Extract TaskTrove tasks with OpenThoughts-Agent tooling:

```bash
cd /home/densemax/work/flavius/surogate/study/OpenThoughts-Agent
python -m scripts.datagen.extract_tasks_from_parquet \
  --parquet open-thoughts/TaskTrove \
  --output_dir /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/tasks \
  --on_exist overwrite
```

Install Harbor CLI:

```bash
uv tool install 'harbor[docker] @ git+https://github.com/marin-community/harbor.git@penfever/working'
```

Run no-model Harbor verifier canary:

```bash
harbor jobs start --yes \
  -p /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/inferredbugs_canary_tasks/inferredbugs-0001 \
  --agent nop \
  --env docker \
  --n-attempts 1 \
  --n-concurrent 1 \
  --job-name fugu_tasktrove_nop_canary \
  --jobs-dir /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/harbor_jobs
```

Run model-backed Harbor canary:

```bash
harbor jobs start --yes \
  -p /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/inferredbugs_canary_tasks/inferredbugs-0001 \
  --agent terminus-2 \
  --model <provider/model> \
  --env docker \
  --n-attempts 1 \
  --n-concurrent 1 \
  --job-name fugu_tasktrove_model_canary \
  --jobs-dir /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/harbor_jobs
```

Stage the TaskTrove inferredbugs canary shard:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="open-thoughts/TaskTrove",
    repo_type="dataset",
    filename="DCAgent__inferredbugs-sandboxes-verifier/tasks.parquet",
    local_dir="/home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/hf",
)
PY

cd /home/densemax/work/flavius/surogate/study/OpenThoughts-Agent
python -m scripts.datagen.extract_tasks_from_parquet \
  --parquet /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/hf/DCAgent__inferredbugs-sandboxes-verifier/tasks.parquet \
  --output_dir /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/inferredbugs_canary_tasks \
  --on_exist overwrite \
  --limit 24

cd /home/densemax/work/flavius/surogate/ultra
.venv/bin/python -m ultra.cli harbor-materialize \
  /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/inferredbugs_canary_tasks \
  --out-jsonl /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/inferredbugs_canary_taskspecs.jsonl \
  --report-out /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/tasktrove_harbor/inferredbugs_canary_report.json \
  --source-name tasktrove_inferredbugs \
  --source-version v3
```
