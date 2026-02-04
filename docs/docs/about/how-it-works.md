# How Surogate Works

Surogate is designed from the ground up for maximum throughput and efficiency when training Large Language Models (LLMs) on modern NVIDIA hardware. Unlike many frameworks that rely heavily on high-level abstractions which introduce overhead, Surogate bridges the gap between Python's flexibility and CUDA's raw performance with a DSL-driven compiler pipeline.

## Core Architecture

### 1. Native C++/CUDA Execution Engine
The heart of Surogate is a high-performance engine written in C++ and CUDA. This allows for:
- **Zero-overhead execution**: Minimized Python-to-CUDA dispatch latency.
- **Custom Kernel Fusions**: Specialized kernels that combine multiple operations (like activation, normalization, and projection) into a single pass, drastically reducing memory bandwidth bottlenecks.
- **Memory Management**: Fine-grained control over GPU memory allocation and reuse.

### 2. DSL + Graph Compiler
Model architectures are defined using a Python decorator-based DSL. The DSL produces explicit forward graphs and serializes them to a JSON IR, which the runtime consumes for execution. This unlocks:

- **Custom architectures** without changing C++/CUDA code
- **Static shape metadata** for planning and verification
- **Save/recompute annotations** for memory-efficient training
- **Stable tensor naming** for deterministic checkpointing

See the [DSL Language Specification](./dsl.md) for details.

### 3. Graph-Based Autodiff
Surogate derives backward graphs ahead of time from the forward graph. Because the backward pass is generated during compilation rather than at runtime, Surogate can:

- Optimize whole-graph execution (fusion, dead code elimination)
- Plan memory across forward and backward passes
- Avoid Python overhead during training

See [Automatic Differentiation (Autodiff)](./automatic-differentiation.md) for details.

### 4. Multi-threaded Scheduler
Surogate uses a custom multi-threaded backend that handles data loading, gradient synchronization, and kernel dispatch in parallel. This ensures that the GPU is never idling while waiting for the next batch of data or for CPU-bound tasks.

### 5. Mixed-Precision & Quantization
Surogate is at the forefront of mixed-precision training:
- **BF16 & FP8**: Native support for modern data types on Hopper (H100/H200) and Blackwell (B200) GPUs.
- **NVFP4**: Cutting-edge support for 4-bit floating point training on Blackwell architectures.
- **Stochastic Rounding**: Used in low-precision training to maintain numerical stability without sacrificing speed.

## Scalability

### Multi-GPU Training
Surogate implements a high-efficiency Distributed Data Parallel (DDP) strategy. Using multi-threading, it overlaps gradient communication with computation, allowing for near-linear scaling across multiple GPUs.

### Multi-Node with Ray
For very large-scale training, Surogate integrates with **Ray** to manage multi-node clusters. This provides a seamless transition from a single workstation to a massive GPU cluster.

## Deployment & Production
Surogate is built for reliability. With deterministic configurations and explicit training "recipes," users can expect predictable results every time.
