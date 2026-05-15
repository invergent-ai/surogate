# Adaptive Training System

Surogate includes a built-in adaptive training system that monitors your training run in real time and provides actionable diagnostics. It detects problems like loss plateaus, gradient explosions, MoE routing collapse, and inefficient compute usage — then tells you exactly what to fix.

Most components are **always-on** with zero configuration. Three optional features can be enabled with a single flag each.

## Quick Start

Add these flags to your training YAML to enable the full adaptive system:

```yaml
# Optional features (always-on components need no config)
auto_lr_reduction: true   # Auto-fix loss spikes and gradient explosions
early_stop: true           # Stop when training converges or diverges
epoch_adjustment: true     # Scale epochs to Chinchilla-optimal token budget
```

That's it. The rest of the system (phase detection, plateau detection, gradient tracking, MoE monitoring, and cross-component intelligence) runs automatically.

## Training Advisor

The **TrainingAdvisor** is the centerpiece of the adaptive system. It correlates signals from all other components to produce recommendations that no single component could generate alone.

For example: LossGuard detects a loss spike and cuts LR. But if MoEMonitor simultaneously shows routing collapse, the real fix is `router_aux_loss_coef`, not LR. The TrainingAdvisor catches this.

### Advisory Rules

The advisor runs 6 cross-signal checks every training step:

#### Rule 1: Plateau with High LR

**Detects:** Loss is plateaued, gradients are flat, but LR is still high.

**What you'll see:**
```
[Advisor] Plateau with flat gradients at step 5000: loss stalled for 50 steps
while LR=2.00e-04 is still high. Consider reducing learning rate or adjusting schedule.
```

**What to do:** Reduce `learning_rate` or switch to a more aggressive schedule decay.

#### Rule 2: Divergence from MoE Routing Collapse

**Detects:** Training is diverging with rising gradients AND MoE routing is unhealthy.

**What you'll see:**
```
[Advisor] MoE routing collapse driving divergence at step 3200: loss is diverging
with rising gradients (trend=+25.00%) AND routing is unhealthy (balance=0.15,
utilization=40%). Fix routing first — increase router_aux_loss_coef rather than reducing LR.
```

**What to do:** Increase `router_aux_loss_coef` (e.g. 0.01 → 0.05). Do NOT reduce LR first.

#### Rule 3: Gradient Vanishing

**Detects:** Training appears to be converging but gradients are shrinking and loss improvement is stalling.

**What you'll see:**
```
[Advisor] Gradient vanishing at step 8000: gradients shrinking (trend=-3.50%)
for 20 steps while loss improvement stalls.
Consider increasing learning rate or reducing weight decay.
```

**What to do:** Try increasing `learning_rate` slightly, or reduce `weight_decay`.

#### Rule 4: Loss Spike Correlated with MoE Issues

**Detects:** LossGuard triggered recently AND MoE expert utilization dropped.

**What you'll see:**
```
[Advisor] Loss spike correlates with MoE routing issues at step 4100: LossGuard
triggered at step 4095 and expert utilization is low (35%).
Routing instability is likely the root cause — increase router_aux_loss_coef before adjusting LR.
```

**What to do:** Increase `router_aux_loss_coef` before trying LR changes.

#### Rule 5: LR Reductions Not Helping

**Detects:** LossGuard has applied 2+ permanent LR reductions but training still isn't converging.

**What you'll see:**
```
[Advisor] LR reductions not helping at step 6000: 2 permanent LR reductions applied
but training is still in plateau phase. The problem may not be learning rate —
check data quality, batch size, or model architecture.
```

**What to do:** The issue is likely not LR. Check your dataset for quality issues, try a different batch size, or reconsider the model architecture.

#### Rule 6: Warmup Too Short

**Detects:** Loss is still dropping steeply when warmup ends (one-shot check).

**What you'll see:**
```
[Advisor] Warmup may be too short: loss still dropping steeply (8.5% over last 10 steps)
as warmup ends at step 200. Consider increasing warmup_steps from 200 to ~300.
```

**What to do:** Increase `warmup_ratio` or `warmup_steps`.

### Cooldown

All advisor rules respect a per-category cooldown (default: 200 steps) to prevent log spam. Each rule can only fire once per cooldown period.

## Training Phases

The **PhaseDetector** classifies your training run into one of five phases by comparing older and recent halves of a rolling loss window:

| Phase          | Meaning                       | What Happens                             |
| -------------- | ----------------------------- | ---------------------------------------- |
| **WARMUP**     | First 50 steps                | Statistics unreliable, no classification |
| **CONVERGING** | Loss steadily decreasing      | Normal healthy training                  |
| **PLATEAU**    | Loss improvement < 0.1%       | Training may be stalled                  |
| **UNSTABLE**   | High loss variance (CV > 15%) | Loss is erratic                          |
| **DIVERGING**  | Loss trending upward          | Training is failing                      |

Phase transitions are logged automatically:

```
Training phase: converging -> plateau at step 5000 (previous phase lasted 3200 steps)
```

The current phase is also included in the `StepMetrics` logged to wandb/Aim as the `phase` field.

**How classification works:** The detector maintains a rolling window (default: 100 steps). It splits the window in half and compares the mean loss of the older half to the recent half:

- Relative improvement > 0.1% → CONVERGING
- Improvement between -1% and 0.1% → PLATEAU
- Improvement < -1% (loss increasing) → DIVERGING
- Coefficient of variation > 15% → UNSTABLE

## Plateau Detection

The **PlateauDetector** specifically watches for loss stagnation and logs a warning when it persists:

```
Plateau detected at step 5000: loss barely improving over last 200 steps
(older_mean=2.3456, recent_mean=2.3412, improvement=0.0019%).
Consider adjusting learning rate or stopping early.
```

It uses patience (3 consecutive checks) and cooldown (200 steps between warnings) to avoid noise. No automatic action is taken.

## Gradient Tracking

The **GradientTracker** maintains a short rolling window (default: 20 steps) of gradient norms and computes:

| Statistic         | Description                                |
| ----------------- | ------------------------------------------ |
| `grad_norm_mean`  | Rolling mean of recent gradient norms      |
| `grad_norm_max`   | Rolling maximum                            |
| `grad_norm_trend` | Slope of linear regression over the window |

These are included in every `StepMetrics` and logged to wandb/Aim.

The tracker warns when gradients are **accelerating** — a sustained upward trend that predicts an explosion before any single step exceeds a threshold:

```
Gradient acceleration at step 3200: grad norms trending upward over last 20 steps
(mean=0.4523, max=0.8901, trend=0.2341, relative=51.77%).
Consider reducing learning rate.
```

The relative trend (trend / mean) makes the threshold scale-independent — it works the same whether your gradient norms are 0.01 or 100.

## Loss Guard (Auto LR Reduction)

**Config:** `auto_lr_reduction: true`

The **LossGuard** detects loss spikes and gradient explosions, then automatically reduces the learning rate using a two-stage escalation:

### Detection Criteria

| Anomaly            | Trigger                                              |
| ------------------ | ---------------------------------------------------- |
| Loss spike         | Loss > rolling mean + 3σ AND absolute change > 0.5   |
| Gradient explosion | Gradient norm > 10× rolling mean OR > 100 (absolute) |
| Non-finite values  | `inf` or `nan` in loss or gradient norm              |

### Escalation Policy

```
Normal → Temporary Override → Temporary Override → Permanent Reduction
         (1st anomaly)        (2nd anomaly)         (3rd anomaly)
```

1. **Temporary override** (first 2 anomalies): LR is cut to 50% of scheduled value, then linearly blends back to the schedule over 50 steps (grace period).
2. **Permanent reduction** (after 2 temporary overrides without recovery): Base LR is permanently scaled by 0.5×. Up to 5 permanent reductions total.

Log output:

```
Auto LR override: loss spike at step 1200 (loss=4.5678, grad_norm=0.89).
LR: 2.00e-04 -> 1.00e-04 (temporary, 50 step grace) [override 1/2]
```

```
Auto LR reduction: gradient explosion at step 2400 (loss=3.2100, grad_norm=145.23).
LR: 2.00e-04 -> 1.00e-04 (permanent) [reduction 1/5]
```

### How the Temporary Override Works

The LR schedule supports temporary overrides that blend back smoothly:

```
LR ─────────┐
             │  ← spike detected, LR cut to 50%
             └──────────────── gradually blend back to schedule
                 50 steps (grace period)
```

This avoids the jarring effect of permanent reductions for transient anomalies.

## MoE Routing Health

The **MoEMonitor** tracks MoE routing metrics over a rolling window and warns when routing degrades. It is always-on but no-ops for dense (non-MoE) models.

### Warning Types

| Warning           | Trigger                    | Log Prefix                       |
| ----------------- | -------------------------- | -------------------------------- |
| Routing imbalance | `load_imbalance` > 3×      | `[MoE] Routing imbalance`        |
| Severe imbalance  | `load_imbalance` > 10×     | `[MoE] Severe routing imbalance` |
| Low utilization   | `expert_utilization` < 80% | `[MoE] Low expert utilization`   |
| Expert collapse   | `expert_utilization` < 50% | `[MoE] Expert collapse risk`     |
| Aux-loss spike    | `aux_loss` > mean + 3σ     | `[MoE] Aux-loss spike`           |

### Routing Diagnostics

The monitor provides a structured health report via `get_routing_diagnostics()`, which the TrainingAdvisor uses for cross-correlation:

| Field               | Range | Meaning                              |
| ------------------- | ----- | ------------------------------------ |
| `healthy`           | bool  | True if no issues detected           |
| `balance_score`     | 0–1   | 1.0 = perfect balance, lower = worse |
| `utilization_score` | 0–1   | Fraction of experts receiving tokens |
| `aux_loss_trend`    | float | Positive = aux-loss increasing (bad) |
| `recommendations`   | list  | Actionable fix suggestions           |

For more details on MoE metrics and tuning, see [Mixture-of-Experts Models](../guides/moe.md).

## Least-Loaded Expert Parallelism (LLEP)

Well-trained MoE models naturally develop imbalanced expert routing — certain experts specialise in specific domains and receive far more tokens than others. This is desirable from a model-quality perspective, but standard Expert Parallelism (EP) is designed around the assumption of balanced load. When routing is skewed, the GPU hosting the busiest experts becomes a bottleneck: latency spikes, memory usage grows, and in extreme cases the overloaded GPU runs out of memory entirely.

**Least-Loaded Expert Parallelism (LLEP)** is Surogate's dynamic load-balancing algorithm for EP training and inference. Instead of altering the model's routing logic (which would change its behaviour), LLEP reroutes *excess tokens* — along with their associated expert weights — from overloaded GPUs to underloaded ones at runtime. The mathematical result is identical to standard EP; only the execution schedule changes.

LLEP activates automatically when `ep_size > 1` and routing imbalance crosses a configurable threshold.

### How LLEP Works

At each MoE layer, LLEP measures the per-GPU token load across the EP group:

1. **Measure**: global expert token counts are collected across all EP ranks.
2. **Check threshold**: if `max_gpu_load / mean_gpu_load` is below `ep_load_balance_threshold`, standard EP runs unchanged (no overhead).
3. **Assign**: the Least-Loaded Assignment (LLA) algorithm redistributes excess tokens from overloaded GPUs to underloaded ones, subject to a per-GPU capacity limit (the `α` factor).
4. **Transfer**: both the token batches and the corresponding expert weight slices are transferred peer-to-peer to the receiving GPU.
5. **Compute**: each GPU runs grouped GEMMs for its native experts plus any spilled experts it received.
6. **Combine**: outputs and gradients are routed back to the originating GPU and accumulated.

LLEP computes the exact MoE forward pass — no approximations, no changes to routing probabilities. Gradient flow is fully supported, making it equally applicable to training and inference.

### Configuration

LLEP is enabled by setting `ep_size > 1`. The threshold controls how aggressively it activates:

```yaml
gpus: 8
ep_size: 4                       # 4-way expert parallelism
ep_load_balance_threshold: 1.3   # activate LLEP when max/mean > 1.3 (default)
```

| Parameter | Type | Default | Effect |
| --- | --- | --- | --- |
| `ep_size` | int | 1 | Number of GPUs per EP group. Must divide `gpus` and `num_experts`. |
| `ep_load_balance_threshold` | float | 1.3 | Imbalance ratio above which LLEP activates. Lower = more aggressive rebalancing. |

**Threshold guidance:**

| Value | Behaviour |
| --- | --- |
| `1.0` | Always active — LPT scheduling runs every layer regardless of balance |
| `1.3` | Default — activates only when meaningful imbalance is present |
| `2.0+` | Conservative — only triggers under severe imbalance |

For post-training and fine-tuning on domain-specific data (where expert specialisation is strongest), `1.0` or `1.3` are recommended. For pre-training where routing is more uniform, `1.3` is a safe default.

### Performance Characteristics

LLEP is most effective when:

- **Imbalance is high**: under 80–95% token concentration into a small number of experts, LLEP achieves 4–6× speedup over standard EP with stable memory usage.
- **Batch size is large**: larger batches saturate per-GPU capacity, making even load distribution more valuable. With smaller batches the communication overhead may outweigh the benefit.
- **Model hidden size is large**: larger GEMMs are more compute-efficient, so the cost of weight transfers is more easily amortised.

Under perfectly balanced routing, LLEP adds minimal overhead and falls back to standard EP automatically when the threshold is not exceeded.

### Relationship to MoEMonitor Warnings

The MoEMonitor and LLEP are complementary: MoEMonitor detects *training-level* routing health issues (collapse, instability), while LLEP handles *system-level* load imbalance transparently.

If you see MoEMonitor warnings while using EP, LLEP context matters:

| Warning | With LLEP active | Recommended action |
| --- | --- | --- |
| `[MoE] Routing imbalance` | Normal — LLEP is redistributing work | No action needed; monitor if imbalance grows |
| `[MoE] Severe routing imbalance` | LLEP may be saturated; very extreme skew | Lower `ep_load_balance_threshold` to `1.0`, check `router_aux_loss_coef` |
| `[MoE] Expert collapse risk` | Routing collapse, not a load problem | Increase `router_aux_loss_coef`; LLEP cannot fix collapse |
| `[Advisor] MoE routing collapse driving divergence` | Training instability | Fix `router_aux_loss_coef` first; LLEP is a throughput tool, not a stability fix |

LLEP is designed to tolerate and exploit natural expert specialisation. It is not a substitute for fixing genuine router collapse — use `router_aux_loss_coef` and `router_z_loss_coef` for that.

### LLEP with QLoRA and Expert Offloading

LLEP is compatible with all QLoRA variants. When combined with `qlora_offload_experts: true`, each GPU offloads only its local expert shard to CPU. Spilled experts are fetched on demand during LLEP redistribution:

```yaml
model: Qwen/Qwen3-30B-A3B
gpus: 4
ep_size: 2
ep_load_balance_threshold: 1.0

lora: true
lora_rank: 16
qlora_fp8: true
qlora_offload_experts: true
```

See [Expert Parallelism](../guides/moe.md#expert-parallelism-ep) for the full EP configuration reference.


## Early Stopping

**Config:** `early_stop: true`

The **EarlyStopping** module monitors four independent criteria and stops training when ANY of them fires:

| Criterion            | Check Frequency | Trigger                                         |
| -------------------- | --------------- | ----------------------------------------------- |
| Convergence score    | Every eval      | Score > 0.85 for 5 consecutive evals            |
| Compute efficiency   | Every step      | Loss reduction per FLOP drops below 50% of peak |
| Sustained divergence | Every step      | DIVERGING phase for 200+ consecutive steps      |
| Sustained plateau    | Every step      | PLATEAU phase for 500+ consecutive steps        |

### Convergence Score

The convergence score combines two signals:

- **Stability** (60% weight): 1 minus the coefficient of variation of the last 5 eval losses. High stability = loss has settled.
- **Improvement rate** (40% weight): How much eval loss improved from the previous eval. No improvement = score goes up.

A score > 0.85 means the model is no longer learning meaningfully. When this persists for 5 consecutive evals, training stops.

### Compute Efficiency

Uses the Chinchilla `6N` approximation (FLOPs/token = 6 × model parameters) to estimate loss reduction per FLOP. When this drops below 50% of the peak efficiency observed during training, further compute is unlikely to help.

Log output:

```
Early stopping at step 15000: compute efficiency collapsed
(current 1.23e-15 < 50% of peak 4.56e-15)
```

## Recommended Corrective Actions

| Log Message                                         | Meaning                           | Recommended Action                                      |
| --------------------------------------------------- | --------------------------------- | ------------------------------------------------------- |
| `Training phase: converging -> plateau`             | Loss stopped improving            | Check if LR is too low or if model has converged        |
| `Training phase: converging -> diverging`           | Loss is increasing                | Reduce LR, check for data issues                        |
| `Training phase: converging -> unstable`            | Loss is erratic                   | Reduce LR, increase batch size, check gradient clipping |
| `Plateau detected at step N`                        | Sustained stagnation              | Consider reducing LR, adjusting schedule, or stopping   |
| `Gradient acceleration at step N`                   | Gradients growing fast            | Reduce LR before explosion occurs                       |
| `Auto LR override: loss spike`                      | Transient anomaly                 | Usually self-resolves with the temporary override       |
| `Auto LR reduction: ... (permanent)`                | Repeated anomalies                | Training may need fundamental changes                   |
| `[MoE] Expert collapse risk`                        | Most tokens to few experts        | Increase `router_aux_loss_coef`, enable `train_router`  |
| `[MoE] Severe routing imbalance`                    | Very uneven expert load           | Increase `router_aux_loss_coef`                         |
| `[MoE] Aux-loss spike`                              | Router destabilising              | Increase `router_z_loss_coef`                           |
| `[Advisor] Plateau with flat gradients`             | LR too high for plateau           | Reduce `learning_rate`                                  |
| `[Advisor] MoE routing collapse driving divergence` | Routing is the root cause         | Fix `router_aux_loss_coef` before LR                    |
| `[Advisor] Gradient vanishing`                      | Gradients shrinking               | Increase LR or reduce `weight_decay`                    |
| `[Advisor] Loss spike correlates with MoE`          | Routing caused the spike          | Fix routing before LR                                   |
| `[Advisor] LR reductions not helping`               | Problem is not LR                 | Check data quality, batch size, architecture            |
| `[Advisor] Warmup may be too short`                 | Loss still dropping at warmup end | Increase `warmup_ratio`                                 |
| `Early stopping at step N: convergence`             | Model converged                   | Training complete                                       |
| `Early stopping at step N: compute efficiency`      | Diminishing returns               | Further training unlikely to help                       |
| `Early stopping at step N: sustained divergence`    | Unrecoverable failure             | Fix hyperparameters and restart                         |
| `Early stopping at step N: sustained plateau`       | Permanently stalled               | Training has stalled, stop and evaluate                 |

## Chinchilla Token Budget (only for Pre-Training)

**Config:** `epoch_adjustment: true`

At training start, Surogate computes the Chinchilla-optimal token budget:

```
tokens_optimal = 20 × model_parameters
```

This is always logged for reference:

```
Chinchilla budget: 12.0B tokens (20 × 600.0M params) | Planned: 8.4B tokens (70.0% of budget)
```

When `epoch_adjustment: true`, Surogate automatically adjusts `num_epochs` so the total tokens match the Chinchilla budget:

```
Epoch adjustment: 3 -> 5 epochs (Chinchilla budget 12.0B tokens, dataset 2.4B tokens)
```

This only applies when `max_steps` is not explicitly set.

## Configuration Reference

### Config Flags

| Option              | Type | Default | Description                                         |
| ------------------- | ---- | ------- | --------------------------------------------------- |
| `auto_lr_reduction` | bool | `false` | Enable LossGuard (automatic LR reduction on spikes) |
| `early_stop`        | bool | `false` | Enable multi-criteria early stopping                |
| `epoch_adjustment`  | bool | `false` | Adjust epochs to Chinchilla-optimal token budget    |


## See Also

- [Training Metrics & Monitoring](../guides/metrics.md) — All metrics logged during training
- [Mixture-of-Experts Models](../guides/moe.md) — MoE-specific configuration and tuning
- [Configuration Reference](../reference/config.md) — Complete YAML config options
