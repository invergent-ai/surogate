"""Source-policy registry (ultra-data2 §3).

Every source is tagged with a policy class that gates which splits its tasks may
land in. This formalizes the router's existing eval-denylist discipline (normalized
prompt-hash denylist + EVAL_ONLY) into typed classes the registry can enforce.
"""

from __future__ import annotations

from .schemas import Split, SourcePolicy

# Which splits each policy permits a task to occupy.
_POLICY_SPLITS: dict[str, set[str]] = {
    "train_allowed": {
        "pool_discovery",
        "pool_validation",
        "grpo_train",
        "online_validation",
        "diagnostic",
    },
    "pool_only": {"pool_discovery", "pool_validation", "diagnostic"},
    "online_validation": {"online_validation", "diagnostic"},
    "final_eval_only": {"final_eval"},
    "diagnostic_only": {"diagnostic"},
}


def policy_allows_split(policy: SourcePolicy, split: Split) -> bool:
    return split in _POLICY_SPLITS[policy]


def is_train_allowed(policy: SourcePolicy) -> bool:
    return policy == "train_allowed"


def allowed_splits(policy: SourcePolicy) -> list[str]:
    """Splits a source under this policy may occupy — used to fill a SourceManifest's allowed_uses."""
    return sorted(_POLICY_SPLITS[policy])


# Recommended per-source policy (ultra-data2 §3). Adapters default to these; a source
# absent here must declare its policy explicitly.
SOURCE_POLICY: dict[str, SourcePolicy] = {
    # agentic / coding
    "existing_bank": "train_allowed",  # router bank: eval families already excluded upstream
    "swe_smith": "train_allowed",
    "swe_bench_verified": "final_eval_only",
    "terminal_bench_official": "final_eval_only",
    "livecodebench_old": "train_allowed",
    "livecodebench_latest": "final_eval_only",
    "bigcodebench": "train_allowed",
    "scicode_dev": "train_allowed",
    "scicode_test": "final_eval_only",
    # math / science / knowledge
    "math_train": "train_allowed",
    "math500": "final_eval_only",
    "aime_old": "train_allowed",
    "aime_2025plus": "final_eval_only",
    "mmlu_pro": "train_allowed",
    "gpqa_diamond": "final_eval_only",
    "hle": "final_eval_only",
    "charxiv": "final_eval_only",
    # tool / long-context
    "tau_custom": "train_allowed",
    "tau3_banking": "final_eval_only",
    "aa_lcr": "final_eval_only",
    "mrcr": "final_eval_only",
    "longctx_generated": "train_allowed",
}

# Single-step bank sources (router-era HF loaders, ported as Ultra adapters).
SOURCE_POLICY.update(
    {
        "numina_math": "train_allowed",
        "omni_math": "train_allowed",
        "code_contests": "train_allowed",
        "taco": "train_allowed",
        "supergpqa": "train_allowed",
        "humaneval": "final_eval_only",  # reported benchmark — held out
        "mbpp": "final_eval_only",  # reported benchmark — held out
    }
)

# Custom / generated sources (no single public dataset; produced upstream).
SOURCE_POLICY.update(
    {
        "github_issue": "train_allowed",
        "terminal_custom": "train_allowed",
        "sequential_sim": "train_allowed",
        "autoresearch": "diagnostic_only",  # expensive, late-stage Ultra tasks
        "role_probe": "diagnostic_only",  # pool-selection diagnostics, not GRPO data
    }
)
