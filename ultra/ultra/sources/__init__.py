"""Source adapters: each emits the canonical ``TaskSpec`` for one source family.

``SOURCE_ADAPTERS`` is the name -> adapter-class registry the ingestion pipeline uses.

Runnable now (route to the existing ``direct_qa`` / ``code_exec`` harnesses): the math,
multiple-choice and code families. EXECUTION-PENDING (emit valid TaskSpecs, but their
harness is not built yet): repo/terminal, tool-dialogue, long-context, vision, simulator,
and the derived role probes.
"""

from .base import RawTaskRef, SourceAdapter, ValidationReport
from .code import (
    BigCodeBenchAdapter,
    CodeContestsAdapter,
    HumanEvalAdapter,
    LiveCodeBenchAdapter,
    LiveCodeBenchProAdapter,
    MBPPAdapter,
    SciCodeAdapter,
    TACOAdapter,
)
from .direct import (
    AIMEAdapter,
    GPQAStyleAdapter,
    HLEStyleAdapter,
    Math500Adapter,
    MathAdapter,
    MMLUProAdapter,
    NuminaMathAdapter,
    OmniMathAdapter,
    SuperGPQAAdapter,
)
from .existing_bank import ExistingBankAdapter
from .harbor import (
    HarborTaskBundleAdapter,
    discover_harbor_task_dirs,
    harbor_task_to_spec,
    materialize_harbor_tasks,
)
from .longcontext import LongContextDocPackAdapter, MRCRStyleAdapter
from .raw import RawRecordAdapter
from .repo import GitHubIssueAdapter, SWEbenchAdapter, SWEsmithAdapter, TerminalBenchAdapter
from .roleprobe import RoleProbeAdapter
from .simulator import AutoResearchAdapter, SequentialSimAdapter
from .tool import TauBenchAdapter
from .vision import CharXivAdapter

SOURCE_ADAPTERS: dict[str, type] = {
    # router bank
    "existing_bank": ExistingBankAdapter,
    # --- runnable: math (direct_qa) ---
    "numina_math": NuminaMathAdapter,
    "math500": Math500Adapter,
    "aime_old": AIMEAdapter,
    "omni_math": OmniMathAdapter,
    # --- runnable: multiple choice (direct_qa) ---
    "mmlu_pro": MMLUProAdapter,
    "supergpqa": SuperGPQAAdapter,
    "gpqa_diamond": GPQAStyleAdapter,
    "hle": HLEStyleAdapter,
    # --- runnable: code (code_exec) ---
    "humaneval": HumanEvalAdapter,
    "mbpp": MBPPAdapter,
    "code_contests": CodeContestsAdapter,
    "taco": TACOAdapter,
    "livecodebench_old": LiveCodeBenchAdapter,
    "livecodebench_latest": LiveCodeBenchProAdapter,
    "bigcodebench": BigCodeBenchAdapter,
    "scicode_dev": SciCodeAdapter,
    # --- execution-pending: repo / terminal (opencode_repo / terminal_sandbox) ---
    "swe_smith": SWEsmithAdapter,
    "swe_bench_verified": SWEbenchAdapter,
    "github_issue": GitHubIssueAdapter,
    "terminal_custom": TerminalBenchAdapter,
    "tasktrove_harbor": HarborTaskBundleAdapter,
    # --- execution-pending: tool / long-context / vision ---
    "tau_custom": TauBenchAdapter,
    "longctx_generated": LongContextDocPackAdapter,
    "mrcr": MRCRStyleAdapter,
    "charxiv": CharXivAdapter,
    # --- execution-pending: simulators ---
    "sequential_sim": SequentialSimAdapter,
    "autoresearch": AutoResearchAdapter,
    # --- derived (diagnostic) ---
    "role_probe": RoleProbeAdapter,
}

__all__ = [
    "SOURCE_ADAPTERS",
    "RawTaskRef",
    "RawRecordAdapter",
    "SourceAdapter",
    "ValidationReport",
    "ExistingBankAdapter",
    "MathAdapter",
    "NuminaMathAdapter",
    "Math500Adapter",
    "AIMEAdapter",
    "OmniMathAdapter",
    "MMLUProAdapter",
    "SuperGPQAAdapter",
    "GPQAStyleAdapter",
    "HLEStyleAdapter",
    "HumanEvalAdapter",
    "MBPPAdapter",
    "CodeContestsAdapter",
    "TACOAdapter",
    "LiveCodeBenchAdapter",
    "LiveCodeBenchProAdapter",
    "BigCodeBenchAdapter",
    "SciCodeAdapter",
    "SWEsmithAdapter",
    "SWEbenchAdapter",
    "GitHubIssueAdapter",
    "TerminalBenchAdapter",
    "HarborTaskBundleAdapter",
    "discover_harbor_task_dirs",
    "harbor_task_to_spec",
    "materialize_harbor_tasks",
    "TauBenchAdapter",
    "LongContextDocPackAdapter",
    "MRCRStyleAdapter",
    "CharXivAdapter",
    "SequentialSimAdapter",
    "AutoResearchAdapter",
    "RoleProbeAdapter",
]
