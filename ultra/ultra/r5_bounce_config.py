"""Regenerate the office bounce orch config from orch_r5.yaml.

orch_r5_bounce_office.yaml = orch_r5.yaml + the agentic office lanes
(tau2 telecom SOLO, CRMArena query) + the tau-retail expansion. It is always
REGENERATED from the main config — the one time it was hand-maintained it
went stale within hours (missing the bird lane). Edit orch_r5.yaml, then
rerun this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "director" / "manifests" / "fugu_clean_v1" / "grpo_pilot_train"

HEADER = """\
# r5 LAUNCH CONFIG — the FULL 10-lane mixture (user-decided 2026-07-28:
# launch with everything from step 0; the two-phase bounce plan is retired
# since all office lanes were ready pre-launch). 6 probe-curated single-turn
# lanes (0.68) + 4 agentic lanes (0.32). Ratios are retention-proportional
# with a 10%% floor (probe: 440 retained of 1,895).
# REGENERATED from orch_r5.yaml by ultra.r5_bounce_config — edit that file,
# then rerun this module.
#
"""

RETAIL_OLD = """\
      task_name: fugu_r5_tool_dialogue
      lane: tool_dialogue
      shuffle: true
      seed: 5104
      pilot_config_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/pilot_config_r5_envlanes.json
      live_safety_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/live_safety_r5_envlanes.json"""

RETAIL_NEW = """\
      task_name: fugu_r5_tool_dialogue
      lane: tool_dialogue
      shuffle: true
      seed: 5104
      pilot_config_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/pilot_config_r5_office.json
      # tau retail expanded 60 -> the full 500-task TRAIN split at this bounce
      # (airline is test-only and stays sealed).
      task_manifest_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/tau_retail_train_full_taskspecs.jsonl
      live_safety_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/live_safety_r5_office.json"""

OFFICE_LANES = """\
  # ---- agentic office lanes (bounced in 2026-07-28) ----
  # telecom: train draws come from telecom_full MINUS the published
  # 114-task benchmark (sealed, tau2_telecom_sealed_eval_ids.json).
  - id: fugu-ultra-pilot
    name: fugu_r5_office_telecom
    path: /home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot
    args:
      task_name: fugu_r5_office_telecom
      lane: office_telecom
      shuffle: true
      seed: 5106
      pilot_config_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/pilot_config_r5_office.json
      task_manifest_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/tau2_telecom_taskspecs.jsonl
      live_safety_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/live_safety_r5_office.json
      provider_mode: live
      allow_yunwu_live: false
      force_step_budget: long
      worker_max_tokens: 8192
      worker_reasoning_effort: high
      max_concurrency: 8
      max_retries: 2
      timeout_s: 600.0
  # crm: exact-match tasks only, 20% stratified holdout SEALED
  # (crmarena_sealed_holdout_taskspecs.jsonl).
  - id: fugu-ultra-pilot
    name: fugu_r5_office_crm
    path: /home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot
    args:
      task_name: fugu_r5_office_crm
      lane: office_crm
      shuffle: true
      seed: 5110
      pilot_config_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/pilot_config_r5_office.json
      task_manifest_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/crmarena_train_taskspecs.jsonl
      live_safety_path: /home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train/live_safety_r5_office.json
      provider_mode: live
      allow_yunwu_live: false
      force_step_budget: long
      worker_max_tokens: 8192
      worker_reasoning_effort: high
      max_concurrency: 8
      max_retries: 2
      timeout_s: 600.0
batch_size:"""

# e2e share at the bounce: tool 0.085 + repo 0.085 + telecom 0.12 + crm 0.11
# math/code/rlpr/bird/finance/dabstep (retention-proportional, 0.68)
# + tool/repo/telecom/crm (0.32). User-decided 2026-07-28.
BOUNCE_RATIOS = "[0.1668, 0.102, 0.0943, 0.0773, 0.1082, 0.1314, 0.08, 0.07, 0.09, 0.08]"


def regenerate(manifest_dir: Path = MANIFEST_DIR) -> dict[str, Any]:
    main = (manifest_dir / "orch_r5.yaml").read_text()
    text = HEADER + main
    if RETAIL_OLD not in text:
        raise RuntimeError("orch_r5.yaml tool_dialogue block changed — update "
                           "ultra.r5_bounce_config templates")
    text = text.replace(RETAIL_OLD, RETAIL_NEW)
    text = text.replace("batch_size:", OFFICE_LANES, 1)

    import re
    match = re.search(r"  env_ratios: \[[^\]]+\]", text)
    if not match:
        raise RuntimeError("env_ratios line not found in orch_r5.yaml")
    text = text.replace(
        match.group(0),
        f"  # FINAL (user-decided 2026-07-28): retention-proportional + 0.32 agentic.\n"
        f"  env_ratios: {BOUNCE_RATIOS}")

    # Every path in the emitted config is ABSOLUTE: the env server resolves
    # relative paths against ITS OWN cwd, so a launch from any directory
    # other than the repo root fails to find the pilot config (2026-07-28).
    text = text.replace(": /home/densemax/work/flavius/surogate/director/manifests/", f": {ROOT}/director/manifests/")
    if ": director/" in text:
        raise RuntimeError("relative director/ paths survived absolutization")

    out = manifest_dir / "orch_r5_full.yaml"
    out.write_text(text)

    import yaml
    cfg = yaml.safe_load(text)
    lanes = [e["name"] for e in cfg["env"]]
    ratios = cfg["buffer"]["env_ratios"]
    if len(lanes) != len(ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise RuntimeError(f"bounce config inconsistent: {len(lanes)} lanes, "
                           f"{len(ratios)} ratios, sum {sum(ratios)}")
    return {"path": str(out), "lanes": lanes, "ratio_sum": round(sum(ratios), 6)}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(regenerate(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
