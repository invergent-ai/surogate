from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn


# Mock LoRA linear layer for testing
class MockLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, out_features, bias=False)})
        self.scaling = {"default": 1.0}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False

        # Initialize weights
        nn.init.kaiming_uniform_(self.base_layer.weight)
        nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        nn.init.zeros_(self.lora_B["default"].weight)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        lora_out = self.lora_B["default"](self.lora_A["default"](x))
        return result + self.scaling["default"] * lora_out


class MockAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, rank: int = 16):
        super().__init__()
        self.q_proj = MockLoRALinear(hidden_dim, hidden_dim, rank)
        self.k_proj = MockLoRALinear(hidden_dim, hidden_dim, rank)
        self.v_proj = MockLoRALinear(hidden_dim, hidden_dim, rank)
        self.o_proj = MockLoRALinear(hidden_dim, hidden_dim, rank)


class MockMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, rank: int = 16):
        super().__init__()
        self.gate_proj = MockLoRALinear(hidden_dim, intermediate_dim, rank)
        self.up_proj = MockLoRALinear(hidden_dim, intermediate_dim, rank)
        self.down_proj = MockLoRALinear(intermediate_dim, hidden_dim, rank)


@dataclass
class BenchmarkConfig:
    batch_size: int
    seq_len: int
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    lora_rank: int
    warmup_iters: int = 10
    benchmark_iters: int = 100
    backward: bool = True


@dataclass
class BenchmarkResult:
    name: str
    config: BenchmarkConfig
    forward_time_ms: float
    backward_time_ms: float | None
    total_time_ms: float
    memory_mb: float


@contextmanager
def cuda_timer():
    """Context manager for accurate CUDA timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    class Timer:
        def elapsed_time(self):
            return start.elapsed_time(end)

    timer = Timer()
    yield timer
    end.record()
    torch.cuda.synchronize()


def benchmark_standard_qkv(attn: MockAttention, x: torch.Tensor, config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark standard QKV implementation."""
    # Warmup
    for _ in range(config.warmup_iters):
        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)
        if config.backward:
            loss = (q.sum() + k.sum() + v.sum())
            loss.backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Forward benchmark
    forward_times = []
    for _ in range(config.benchmark_iters):
        x_clone = x.clone().requires_grad_(config.backward)
        with cuda_timer() as timer:
            q = attn.q_proj(x_clone)
            k = attn.k_proj(x_clone)
            v = attn.v_proj(x_clone)
        torch.cuda.synchronize()
        forward_times.append(timer.elapsed_time())

    # Backward benchmark
    backward_times = []
    if config.backward:
        for _ in range(config.benchmark_iters):
            x_clone = x.clone().requires_grad_(True)
            q = attn.q_proj(x_clone)
            k = attn.k_proj(x_clone)
            v = attn.v_proj(x_clone)
            loss = q.sum() + k.sum() + v.sum()
            with cuda_timer() as timer:
                loss.backward()
            torch.cuda.synchronize()
            backward_times.append(timer.elapsed_time())

    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    fwd_avg = sum(forward_times) / len(forward_times)
    bwd_avg = sum(backward_times) / len(backward_times) if backward_times else None

    return BenchmarkResult(
        name="standard_qkv",
        config=config,
        forward_time_ms=fwd_avg,
        backward_time_ms=bwd_avg,
        total_time_ms=fwd_avg + (bwd_avg or 0),
        memory_mb=memory_mb,
    )


def benchmark_fused_qkv(attn: MockAttention, x: torch.Tensor, config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark fused LoRA QKV implementation."""
    from surogate.train.lora_fast.fast_kernels import apply_lora_qkv
    import types

    # Bind the fused method
    attn.apply_qkv = types.MethodType(apply_lora_qkv, attn)

    # Warmup
    for _ in range(config.warmup_iters):
        q, k, v = attn.apply_qkv(x)
        if config.backward:
            loss = q.sum() + k.sum() + v.sum()
            loss.backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Forward benchmark
    forward_times = []
    for _ in range(config.benchmark_iters):
        x_clone = x.clone().requires_grad_(config.backward)
        with cuda_timer() as timer:
            q, k, v = attn.apply_qkv(x_clone)
        torch.cuda.synchronize()
        forward_times.append(timer.elapsed_time())

    # Backward benchmark
    backward_times = []
    if config.backward:
        for _ in range(config.benchmark_iters):
            x_clone = x.clone().requires_grad_(True)
            q, k, v = attn.apply_qkv(x_clone)
            loss = q.sum() + k.sum() + v.sum()
            with cuda_timer() as timer:
                loss.backward()
            torch.cuda.synchronize()
            backward_times.append(timer.elapsed_time())

    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    fwd_avg = sum(forward_times) / len(forward_times)
    bwd_avg = sum(backward_times) / len(backward_times) if backward_times else None

    return BenchmarkResult(
        name="fused_qkv",
        config=config,
        forward_time_ms=fwd_avg,
        backward_time_ms=bwd_avg,
        total_time_ms=fwd_avg + (bwd_avg or 0),
        memory_mb=memory_mb,
    )


def benchmark_standard_mlp(mlp: MockMLP, x: torch.Tensor, config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark standard MLP with SwiGLU."""
    def standard_swiglu_mlp(mlp: MockMLP, x: torch.Tensor) -> torch.Tensor:
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        hidden = torch.nn.functional.silu(gate) * up
        return mlp.down_proj(hidden)

    # Warmup
    for _ in range(config.warmup_iters):
        out = standard_swiglu_mlp(mlp, x)
        if config.backward:
            out.sum().backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    forward_times = []
    for _ in range(config.benchmark_iters):
        x_clone = x.clone().requires_grad_(config.backward)
        with cuda_timer() as timer:
            out = standard_swiglu_mlp(mlp, x_clone)
        torch.cuda.synchronize()
        forward_times.append(timer.elapsed_time())

    backward_times = []
    if config.backward:
        for _ in range(config.benchmark_iters):
            x_clone = x.clone().requires_grad_(True)
            out = standard_swiglu_mlp(mlp, x_clone)
            with cuda_timer() as timer:
                out.sum().backward()
            torch.cuda.synchronize()
            backward_times.append(timer.elapsed_time())

    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    fwd_avg = sum(forward_times) / len(forward_times)
    bwd_avg = sum(backward_times) / len(backward_times) if backward_times else None

    return BenchmarkResult(
        name="standard_mlp",
        config=config,
        forward_time_ms=fwd_avg,
        backward_time_ms=bwd_avg,
        total_time_ms=fwd_avg + (bwd_avg or 0),
        memory_mb=memory_mb,
    )


def benchmark_fused_mlp(mlp: MockMLP, x: torch.Tensor, config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark fused LoRA MLP with SwiGLU."""
    from surogate.train.lora_fast.fast_kernels import apply_lora_mlp_swiglu
    import types

    mlp.forward = types.MethodType(apply_lora_mlp_swiglu, mlp)

    # Warmup
    for _ in range(config.warmup_iters):
        out = mlp(x)
        if config.backward:
            out.sum().backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    forward_times = []
    for _ in range(config.benchmark_iters):
        x_clone = x.clone().requires_grad_(config.backward)
        with cuda_timer() as timer:
            out = mlp(x_clone)
        torch.cuda.synchronize()
        forward_times.append(timer.elapsed_time())

    backward_times = []
    if config.backward:
        for _ in range(config.benchmark_iters):
            x_clone = x.clone().requires_grad_(True)
            out = mlp(x_clone)
            with cuda_timer() as timer:
                out.sum().backward()
            torch.cuda.synchronize()
            backward_times.append(timer.elapsed_time())

    memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    fwd_avg = sum(forward_times) / len(forward_times)
    bwd_avg = sum(backward_times) / len(backward_times) if backward_times else None

    return BenchmarkResult(
        name="fused_mlp",
        config=config,
        forward_time_ms=fwd_avg,
        backward_time_ms=bwd_avg,
        total_time_ms=fwd_avg + (bwd_avg or 0),
        memory_mb=memory_mb,
    )


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print(f"{'Name':<20} {'Batch':<8} {'SeqLen':<8} {'Hidden':<8} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Mem (MB)':<10}")
    print("=" * 100)

    for r in results:
        bwd_str = f"{r.backward_time_ms:.3f}" if r.backward_time_ms else "N/A"
        print(f"{r.name:<20} {r.config.batch_size:<8} {r.config.seq_len:<8} {r.config.hidden_dim:<8} "
              f"{r.forward_time_ms:<12.3f} {bwd_str:<12} {r.total_time_ms:<12.3f} {r.memory_mb:<10.2f}")


def run_benchmark():
    """Run full benchmark suite."""
    assert torch.cuda.is_available(), "CUDA required for benchmarking"

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Test configurations - vary batch*seq_len to find crossover point
    configs = [
        BenchmarkConfig(batch_size=1, seq_len=512, hidden_dim=4096, intermediate_dim=11008, num_heads=32, lora_rank=16),
        BenchmarkConfig(batch_size=1, seq_len=2048, hidden_dim=4096, intermediate_dim=11008, num_heads=32, lora_rank=16),
        BenchmarkConfig(batch_size=2, seq_len=2048, hidden_dim=4096, intermediate_dim=11008, num_heads=32, lora_rank=16),
        BenchmarkConfig(batch_size=4, seq_len=2048, hidden_dim=4096, intermediate_dim=11008, num_heads=32, lora_rank=16),
        BenchmarkConfig(batch_size=8, seq_len=2048, hidden_dim=4096, intermediate_dim=11008, num_heads=32, lora_rank=16),
    ]

    all_results = []

    for config in configs:
        print(f"\nBenchmarking: batch={config.batch_size}, seq_len={config.seq_len}, hidden={config.hidden_dim}")

        x = torch.randn(config.batch_size, config.seq_len, config.hidden_dim, device=device, dtype=dtype)

        # QKV benchmarks
        attn = MockAttention(config.hidden_dim, config.num_heads, config.lora_rank).to(device, dtype)
        all_results.append(benchmark_standard_qkv(attn, x, config))

        attn = MockAttention(config.hidden_dim, config.num_heads, config.lora_rank).to(device, dtype)
        all_results.append(benchmark_fused_qkv(attn, x, config))

        # MLP benchmarks
        mlp = MockMLP(config.hidden_dim, config.intermediate_dim, config.lora_rank).to(device, dtype)
        all_results.append(benchmark_standard_mlp(mlp, x, config))

        mlp = MockMLP(config.hidden_dim, config.intermediate_dim, config.lora_rank).to(device, dtype)
        all_results.append(benchmark_fused_mlp(mlp, x, config))

        torch.cuda.empty_cache()

    print_results(all_results)


if __name__ == "__main__":
    run_benchmark()
