"""Agentic (multi-turn, tool-using) orchestration for benchmarks like SWE-Bench.

Unlike the single-step SFT track, agentic tasks are multi-turn rollouts in a harness
with a terminal reward. The router selects which worker writes each turn's action
(per-step routing); sep-CMA-ES optimizes that routing against terminal reward.
"""
