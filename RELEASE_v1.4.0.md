# v1.4.0 — ready to push

Everything is committed, merged and tagged locally. Only the network step remains,
which needs GitHub credentials this session does not have.

## Finish the release

    gh auth login -h github.com          # stored token is invalid; keyring holds none
    git push origin main
    git push origin v1.4.0

If you would rather review a PR than take the local merge:

    git push -u origin feat/turnopd-depth-budgeting
    gh pr create --base main --head feat/turnopd-depth-budgeting \
      --title "feat(grpo): turn-aware rollout-depth budgeting + dispatch-PP for GRPO"
    # then reset main to origin/main and merge via the PR:
    #   git checkout main && git reset --hard origin/main

Local state:
  main       d09e12b6  (merge commit)
  tag        v1.4.0 -> d09e12b6
  branch     feat/turnopd-depth-budgeting @ 1300a7c9 (intact, for the PR flow)
  unpushed   1 feature commit + 1 merge commit + 1 tag

## What is in it

Adaptive rollout-depth budgeting (TurnOPD arXiv:2607.05804 §5.1, coverage arm)
  - 1.36x wall clock (462.8s -> 341.4s over 30 steps), reward parity (+0.009 on
    the last 10 steps, i.e. within noise), mean max-turn 8.77 -> 6.63
  - off by default: rollout_depth.enabled; no-op for single-turn envs

Turn-resolved supervision diagnostics
  - turn_ids plumbed orchestrator -> transport -> packer -> trainer
  - turn_diagnostics mode: no C++ change, no extra forward pass

dispatch-PP support for GRPO-style objectives
  - custom per-token gradient seeds replace cross-entropy at the loss stage
  - staged forward that runs THROUGH the loss stage to return logprobs

Framework fixes (all previously blocking, not merely degraded)
  - vLLM 0.25.1: GRPO inference did not start at all
  - TeacherModelConfig: on-policy distillation could never run
  - eager wandb imports took down servers that do not use wandb

## Explicitly NOT adopted

The paper's headline result -- progressive uniform-1/T turn normalization. Our
measured per-turn |KL| peaks at turn 3 (the decision depth), not turn 0, and the
deepest turns are degenerate loops where student and teacher agree. Uniform
weighting would move budget toward exactly those turns. See
project_turnopd_assessment in the project memory for the numbers.

## Test status

  dispatch-PP        32/32
  GRPO              105/105
  full suite        229 passed, 51 skipped

Two failures remain and are PRE-EXISTING, reproduced on a baseline rebuild with
this branch's C++ stashed:
  tests/test_onboarding_qwen3_5.py::TestQwen35OnboardingBackward::test_selected_gradients
  tests/test_onboarding_laguna.py::TestLagunaOnboarding::test_final_norm_output  (missing weights)

## Known sharp edge

dispatch-PP leaves per-step executor state that the next dispatch call clears on
entry. Issuing a NON-dispatch forward (forward_for_grpo, validate) on the same
trainer in between hits an async CUDA launch failure. GRPO alternates staged
forward and fused step, so it never does; documented as a contract note in
tests/train/dispatch_pp/test_custom_dloss.py.
