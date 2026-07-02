# fugu-ultra-pilot

Verifiers environment for the first Fugu-Ultra GRPO pilot.

The model being trained is the Conductor. Each rollout asks it to emit workflow
JSON. The rubric parses the workflow, applies the lane-local worker mask from
`pilot_config.json`, executes the workflow with Ultra harnesses, and returns the
faithful Ultra reward.

Use from Surogate `orch.yaml`:

```yaml
env:
  - id: fugu-ultra-pilot
    path: ./environments/fugu-ultra-pilot
    args:
      pilot_config_path: ./director/manifests/fugu_clean_v1/grpo_pilot_train/pilot_config.json
      task_manifest_path: ./director/manifests/fugu_clean_v1/grpo_pilot_train/taskspecs.jsonl
      artifact_dir: ./director/manifests/fugu_clean_v1/grpo_pilot_train/rollout_artifacts
      provider_mode: live
```
