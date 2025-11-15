# surogate/eval/benchmarks/__init__.py
"""Benchmark evaluation module."""

from .base import BaseBenchmark, BenchmarkConfig, BenchmarkResult
from .registry import BenchmarkRegistry
from .generic import GenericBenchmark

# Auto-register all EvalScope benchmarks
EVALSCOPE_BENCHMARKS = [
    # Standard LLM benchmarks
    'mmlu', 'cmmlu', 'c_eval', 'gsm8k', 'arc', 'arc_challenge', 'arc_easy',
    'hellaswag', 'truthfulqa', 'winogrande', 'bbh', 'humaneval', 'mbpp',
    'boolq', 'drop', 'squad', 'lambada', 'logiqa', 'mathqa', 'ifeval',

    # Math & reasoning
    'math', 'aime', 'aime_2024', 'aime_2025',

    # Specialized
    'super_gpqa', 'longbench', 'longbench_write',

    # Agent & tool use
    'tau_bench', 'tau2_bench', 'toolbench', 'bfcl', 'bfcl_v3', 'bfcl_v4',

    # Custom/General formats
    'general_qa', 'general_mcq',

    # Multimodal/Vision benchmarks
    'mmmu', 'mmmu_pro', 'mathvista', 'math_vista', 'chartqa', 'docvqa',
    'infovqa', 'ai2d', 'seed_bench', 'mm_bench', 'mm_star', 'pope',
    'real_world_qa',
]

# Register all benchmarks using generic class
for benchmark_name in EVALSCOPE_BENCHMARKS:
    # Create a class dynamically
    BenchmarkRegistry._benchmarks[benchmark_name] = GenericBenchmark

__all__ = [
    'BaseBenchmark',
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkRegistry',
    'GenericBenchmark',
]