# Softmax Attention 组织与迁移规范

本文定义 NInfer Softmax Attention 的目标语义、公共契约、源码组织和现有实现迁移方式。
它是 [`op-development.md`](op-development.md) 在 Attention 领域的细化规范。本文描述的是迁移
完成后的唯一目标态；在迁移完成前，源码中仍可见的 `gqa_attention`、`vision_attention`、
`bidirectional_gqa_attention` 和 `swa` 是待替换的现状，不是需要兼容的第二套接口。

## 1. 设计结论

Softmax Attention 与现有 Linear Attention 是两个平行的算法类别：

```text
src/ops/
├── softmax_attention/
└── linear_attention/
```

Softmax Attention 内部按完整数学变换和显式状态效果划分 family，不按模型、模态、Head
拓扑或 CUDA 算法划分：

```text
src/ops/softmax_attention/
├── dense/
└── sliding_window/
```

当前迁移遵循以下结论：

1. MHA、MQA 和 GQA 不是三个 Op，也不是三个源码目录。它们是统一 Q/K/V Attention 的
   Head geometry 取值。
2. Text causal cache Attention、packed Vision Attention 和 DFlash context Attention 同属
   `dense` family，但它们的可见域、物理输入和状态效果不同，保留效果明确的独立 entry。
3. `vision` 是调用方身份，不是 Attention 语义。公共符号、文件名、测试和 benchmark 中不再
   使用 `vision_attention`。
4. `gqa` 只描述 Query Head 到 KV Head 的映射。公共符号、文件名、测试和 benchmark 中不再
   使用 `gqa_attention`。
5. 当前 SWA 使用循环 KV cache、绝对位置和对称局部可见域，是独立的
   `sliding_window` family；公共全名为 `sliding_window_attention`。
6. FlashAttention 是 exact Softmax Attention 的私有实现策略，PagedAttention 是 KV
   存储和寻址策略。二者都不是公共 semantic family。
7. MLA、结构化稀疏 Attention、二维窗口 Attention 和 Deformable Attention 只有在注册目标
   实际需要时才创建 peer family；当前不创建空目录、占位接口或通用注册框架。

## 2. 分类轴

Attention 相关术语必须先归入以下四个互相独立的轴，不能把不同轴的名字并列成目录：

| 轴 | 回答的问题 | 当前例子 | 对源码组织的影响 |
|---|---|---|---|
| 数学 family | 输出公式、可见域和状态转移是什么 | dense、sliding window | 决定 `softmax_attention/` 下的 family |
| Head geometry | 每个 Query Head 使用哪个 KV Head | MHA、MQA、GQA | 普通值和验证规则，不创建 family |
| 输入/状态表示 | K/V 是普通 Tensor、packed segment、线性 cache、循环 cache 还是 latent cache | packed、causal cache、context+query、未来 MLA | 在 family 内形成效果明确的 entry 或新的 family |
| 实现策略 | 如何分块、搬运、归约和寻址 | flash、split-KV、paged、small-T | 只存在于私有 launcher、plan 和 kernel |

以下分类是强制的：

- self-attention 与 cross-attention 在 Q/K/V 已经显式给出后使用同一个 dense 公式；它们不因
  Q/K/V 的来源不同而成为两个 Op。
- causal、非 causal、packed block-diagonal、context+query 和 sliding window 描述可见域或
  状态效果，必须在 entry 契约中明确，不能藏在调用方约定里。
- `prefill`、`decode`、`small_t`、`flash`、`split_kv` 是实现或测量术语，不能成为公共
  semantic entry。
- Text、MTP、Vision、DFlash 是调用位置或模型调度概念，不能出现在通用 Attention Op 名称
  中。

## 3. 统一数学语义与 Head geometry

### 3.1 Head geometry

显式 Q/K/V Softmax Attention 使用一个窄的 host value 表达 Head geometry：

```cpp
struct AttentionHeadGeometry {
    std::int32_t head_dim;
    std::int32_t query_heads;
    std::int32_t kv_heads;
};
```

当前公共表示保持 Q、K、V 的 Head dimension 相同。合法 geometry 满足：

```text
D = head_dim > 0
Hq = query_heads > 0
Hkv = kv_heads > 0
Hq % Hkv == 0
group = Hq / Hkv
kv_head(h) = floor(h / group)
```

逻辑 Tensor 形状为：

```text
Q   [D, Hq,  Tq]
K   [D, Hkv, Tk]
V   [D, Hkv, Tk]
Out [D, Hq,  Tq]
```

geometry 给通用 `Tensor` 轴赋予语义，并供 workspace capacity 查询使用；执行 wrapper 必须
验证 geometry 与 Q/K/V/Out/cache view 一致。它不包含目标 key、模型角色、执行阶段、kernel
选择或设备资源。

MHA、MQA 和 GQA 只是以下三个取值区域：

```text
MHA: Hq == Hkv
MQA: Hkv == 1
GQA: 1 < Hkv < Hq
```

因此不定义 `GqaGeometry`、`GqaExecutionEnvelope`、`gqa_*` launcher 或 `gqa_*` kernel。
若未来真实目标需要非均匀或非连续 Head 映射，应为该数学语义定义新的显式映射契约；不得先
把任意映射 Tensor 加入当前统一 geometry。

### 3.2 Softmax 公式

对 Query token `i`、Query Head `h` 和 entry 定义的可见 Key 集合 `A(i)`：

```text
kh = floor(h / (Hq / Hkv))
score(i,h,j) = scale * dot(Q[:,h,i], K[:,kh,j]), j ∈ A(i)
p(i,h,:)     = stable_softmax(score(i,h,:))
ideal[:,h,i] = Σ(j ∈ A(i)) p(i,h,j) * V[:,kh,j]
```

公共输入 storage boundary 之后的逻辑值进入同一个朴素 FP64 oracle。实现可以使用在线
Softmax、Flash tiling、split-KV、私有低精度 staging 或不同归约树，但不得改变可见集合、cache
编码、公开输出和状态效果。

`scale` 是显式语义参数。当前注册 domain 可只接受对应 geometry 的 `1/sqrt(D)`，但不能把
scale 隐藏在 Vision、Text 或某个 kernel 名称中。

### 3.3 当前注册 geometry

迁移不借机扩大支持面。各 entry 只接收当前生产路径已经支持的有限 domain：

| 使用位置 | geometry | cache/布局 | 迁移后 entry |
|---|---:|---|---|
| Qwen3.6-27B Text/MTP | `D256/Hq24/Hkv4` | BF16 或 INT8-G64 线性 cache | `causal_softmax_attention` |
| Qwen3.6-35B-A3B Text/MTP | `D256/Hq16/Hkv2` | BF16 或 INT8-G64 线性 cache | `causal_softmax_attention` |
| Qwen3.6 Vision | `D72/Hq16/Hkv16` | packed BF16 Q/K/V | `packed_softmax_attention` |
| Qwen3.6-35B-A3B DFlash full | `D128/Hq32/Hkv8` | 只读 BF16 context + query K/V | `context_softmax_attention` |
| Qwen3.6-35B-A3B DFlash local | `D128/Hq32/Hkv8` | 只读 BF16 cyclic context + query K/V | `sliding_window_attention` |

## 4. 公共契约

目标态使用一个共享值 header 和三个公共 contract header：

```text
include/ninfer/ops/attention_geometry.h
include/ninfer/ops/softmax_attention.h
include/ninfer/ops/sliding_window_attention.h
include/ninfer/ops/kv_cache_append.h
```

`attention_geometry.h` 只拥有 `AttentionHeadGeometry` 及其 host-side 合法性规则，使 dense
和 sliding-window contract 不需要互相 include。`softmax_attention.h` 拥有 dense family 的
公式、entry-specific execution envelope 和 workspace capacity 查询。
`sliding_window_attention.h` 拥有局部窗口的独立可见域与循环 cache 契约。
`kv_cache_append.h` 拥有不计算 Attention 的 KV cache 状态转移。

### 4.1 普通 dense Q/K/V

普通单段 Q/K/V 入口采用所期望的直接形式：

```cpp
void softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                       AttentionHeadGeometry geometry, float scale,
                       WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);
```

公式和接口不把 `Tq` 与 `Tk` 强制为同一个语义轴，因此未来 dense cross-attention 不需要新建
family。当前迁移只注册已有的 `D72/Hq16/Hkv16`、`Tq=Tk` 单段 self-attention domain；不能
因为公式一般就宣称任意 shape 或 cross-attention 已支持。

当前迁移可由 packed 实现的单 segment route 承担该入口，不增加第二套 kernel。

### 4.2 Packed dense Q/K/V

当前 Vision Attention 迁移为与模态无关的 packed entry：

```cpp
void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale,
                              const Tensor& cu_seqlens, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream);

void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale,
                              std::int32_t segment_length, Tensor& out,
                              cudaStream_t stream);
```

每个 segment 是独立的 dense non-causal self-attention；不同 segment 之间不可见。第一种
entry 接受严格递增的 device `cu_seqlens`，第二种 entry 表示等长连续 segment。其 workspace
查询改名为：

```cpp
packed_softmax_attention_workspace_capacity_bytes(...)
```

迁移后 contract 中使用 `T` 或 `tokens` 表示通用 packed token 数，不使用 `patches` 作为 Op
层轴名。Vision 调度仍可在自己的代码中把该值称为 `patches`。

### 4.3 Causal cached dense Attention

Text/MTP 的线性 cache Attention 保留三个效果明确的行为：

```cpp
struct CausalAttentionExecutionEnvelope {
    std::uint32_t min_visible_keys;
    std::uint32_t max_visible_keys;
};

void causal_softmax_attention(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    AttentionHeadGeometry geometry, float scale, KVCacheLayerView cache,
    CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
    Tensor& out, cudaStream_t stream);

void causal_softmax_attention_cached(
    const Tensor& q, const Tensor& positions, AttentionHeadGeometry geometry,
    float scale, const KVCacheLayerView& cache,
    CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
    Tensor& out, cudaStream_t stream);
```

`causal_softmax_attention` 先把当前 K/V 写入绝对 `positions` 对应的 cache rows，再让 Query
位置 `p` 看见已经填充的 `[0,p]`。`causal_softmax_attention_cached` 不接受新 K/V，不修改
cache，只读取相同可见域。二者共享同一个数学 oracle和 cache codec 定义，但必须分别直接
qualification。

capacity 查询为：

```cpp
causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, DType cache_dtype,
    CausalAttentionExecutionEnvelope envelope,
    std::int32_t min_tokens, std::int32_t max_tokens);
```

`positions` 决定精确因果可见域；execution envelope 仅保证 graph-safe launch capacity，不是
mask 参数。

### 4.4 Context + query dense Attention

当前 `bidirectional_gqa_attention` 的真实语义是：Query 同时看见只读 persistent context 和
完整 query K/V segment。它迁移为：

```cpp
struct ContextAttentionExecutionEnvelope {
    std::uint32_t min_context;
    std::uint32_t max_context;
};

void context_softmax_attention(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v,
    const Tensor& context_length, AttentionHeadGeometry geometry, float scale,
    const KVCacheLayerView& context, ContextAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);
```

所有 Query rows 都看见 `[0, context_length)` 和完整 query segment；不存在 causal triangle。
`context` 与 query K/V 保持两个物理 segment，不要求拼接或复制。capacity 查询为
`context_softmax_attention_workspace_capacity_bytes(...)`。

`bidirectional` 不再出现在名称中，因为它只描述当前 entry 的可见域；`GQA` 不再出现在名称
中，因为 `D128/Hq32/Hkv8` 已由 geometry 表达。

### 4.5 Sliding-window Attention

SWA 迁移到独立 header 和 family：

```cpp
struct SlidingWindowAttentionExecutionEnvelope {
    std::uint32_t min_context;
    std::uint32_t max_context;
};

void sliding_window_attention(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v,
    const Tensor& positions, AttentionHeadGeometry geometry,
    std::uint32_t window, float scale,
    const CyclicKVCacheLayerView& context,
    SlidingWindowAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);
```

当前唯一注册的 `window` 是 4096。显式参数使 `abs(key_position-query_position) < window` 成为
完整 Op 语义；wrapper 仍只接收已经实现和 qualification 的值。循环物理槽
`absolute_position % window` 是该 entry 的状态布局契约。

公共符号完整拼写 `sliding_window_attention`。`SWA` 可用于论文术语、局部变量或 profiler
说明，但不再作为 contract、文件、测试 target 或 benchmark target 的主名称。

### 4.6 KV cache append 移出 Attention

当前 `gqa_kv_append` 只做 cache 写入和可选 INT8-G64 编码，不计算 Attention。它不是 dense
Attention entry，迁移为 `kv_cache_append`：

```cpp
void kv_cache_append(const Tensor& k, const Tensor& v,
                     const Tensor& positions, KVCacheLayerView cache,
                     cudaStream_t stream);
```

它与现有 `kv_cache_append_prefix` 组成一个 cache state-transition overload group，共同迁移
到：

```text
include/ninfer/ops/kv_cache_append.h
src/ops/kv_cache/append/
```

`kv_cache_append` 覆盖所有输入 rows，并按目标 cache dtype 执行 BF16 exact copy 或
INT8-G64 encode；`kv_cache_append_prefix` 继续接受 device `commit_count`，只写入被接纳前缀。
二者不因被 Attention 调用而归属 `softmax_attention/`。

`causal_softmax_attention` 的 fused append-and-attend entry 必须引用同一 cache codec 语义并
产生相同 code/scale bits；它可以拥有性能所需的融合实现，不必通过公共
`kv_cache_append` 产生额外 launch。

### 4.7 明确不引入的统一接口

不引入以下设计：

- `AttentionOptions`、runtime `AttentionKind`、字符串 registry 或 Op 基类；
- 一个接受任意 dense mask Tensor 的万能入口；
- 用 `is_vision`、`is_decode`、`is_gqa`、`use_flash` 等布尔值选择语义；
- 为 MHA、MQA、GQA 分别提供 overload；
- 把 linear、cyclic、paged、latent cache 塞进一个 `std::variant`；
- 由 wrapper 根据目标 key、layer role 或 Program phase 选择行为。

当一个新目标需要新的完整可见域或状态表示时，先写出闭合公式和状态效果，再决定扩展现有
entry 还是创建 peer family。

## 5. 目标源码组织

迁移完成后的目录如下。文件名表达语义 entry 或真实私有实现路线，不表达调用模型：

```text
src/ops/
├── softmax_attention/
│   ├── common/
│   │   ├── head_mapping.cuh
│   │   └── context_query.cuh
│   ├── dense/
│   │   ├── causal_cache/
│   │   │   ├── causal_softmax_attention.cpp
│   │   │   ├── launch.h
│   │   │   ├── small_t.cu
│   │   │   ├── small_t.cuh
│   │   │   ├── small_t_bf16.cuh
│   │   │   ├── small_t_i8.cuh
│   │   │   ├── prompt.cu
│   │   │   ├── prompt_common.cuh
│   │   │   ├── prompt_bf16.cuh
│   │   │   └── prompt_i8.cuh
│   │   ├── packed/
│   │   │   ├── packed_softmax_attention.cpp
│   │   │   ├── launch.h
│   │   │   ├── launch.cu
│   │   │   └── kernel.cuh
│   │   └── context/
│   │       ├── context_softmax_attention.cpp
│   │       ├── launch.h
│   │       ├── launch.cu
│   │       └── kernel.cuh
│   └── sliding_window/
│       ├── sliding_window_attention.cpp
│       ├── launch.h
│       ├── launch.cu
│       └── kernel.cuh
└── kv_cache/
    ├── codec.cuh
    └── append/
        ├── kv_cache_append.cpp
        ├── launch.h
        ├── launch.cu
        └── kernel.cuh
```

这是责任和归属结构，不要求为了对齐树形而制造空文件：

- entry `.cpp` 负责 contract validation、workspace scope 和有限 dispatch；
- `launch.h`、`launch.cu` 负责私有 plan、grid、block、shared memory 和 launch error；
- `.cuh` 负责 kernel 和 entry-local device computation；
- `softmax_attention/common/` 只接收至少两个 family 已经实际共享的窄 device primitive。

当前 `bidirectional_gqa_attention.cuh` 同时承载 linear-context 与 cyclic-SWA 的共享 device
body。迁移时应把这部分改成无 GQA/SWA 身份的 `common/context_query.cuh`，由
`dense/context` 和 `sliding_window` 各自的 entry kernel 调用。不得让
`sliding_window/` include `dense/context` 的私有 kernel，也不得建立一个通用 Attention
backend 框架。

`head_mapping.cuh` 只实现由 `AttentionHeadGeometry` 已验证事实导出的零成本索引；它不做
runtime registry 或目标选择。

INT8-G64 cache 的 exact encode/decode device primitive 归属 `kv_cache/codec.cuh`。Standalone
append 和 causal append-and-attend 的融合实现都可以复用它；不得在两个 family 内复制 codec，
也不得让 `kv_cache/append/` 反向 include Attention 私有文件。

## 6. 现有契约与符号迁移

### 6.1 公共 header 和符号

| 现有内容 | 目标内容 | 处理 |
|---|---|---|
| （新增共享类型） | `include/ninfer/ops/attention_geometry.h` 中的 `AttentionHeadGeometry` | 只表达 Head 数值事实，不建立 family 或 registry |
| `include/ninfer/ops/gqa_attention.h` | `include/ninfer/ops/softmax_attention.h` | 删除旧 header，不保留 forwarding include |
| `GqaExecutionEnvelope` | `CausalAttentionExecutionEnvelope` | 重命名并去除 GQA 身份 |
| `gqa_attention_workspace_capacity_bytes` | `causal_softmax_attention_workspace_capacity_bytes` | 增加显式 geometry |
| `gqa_attention` | `causal_softmax_attention` | 保留 append-and-attend 状态效果 |
| `gqa_attention_cached` | `causal_softmax_attention_cached` | 保留只读 cache 效果 |
| `gqa_kv_append` | `kv_cache_append` | 移出 Attention family |
| `include/ninfer/ops/vision_attention.h` | `include/ninfer/ops/softmax_attention.h` | 删除 Vision 身份 |
| `vision_attention_workspace_capacity_bytes` | `packed_softmax_attention_workspace_capacity_bytes` | 轴名改为通用 tokens/segments |
| `vision_attention(..., cu_seqlens, ...)` | `packed_softmax_attention(..., cu_seqlens, ...)` | 增加 geometry 与显式 scale |
| `vision_attention(..., segment_length, ...)` | `packed_softmax_attention(..., segment_length, ...)` | 保留等长 segment overload |
| `include/ninfer/ops/bidirectional_gqa_attention.h` | `include/ninfer/ops/softmax_attention.h` | 删除旧 header |
| `GqaContextExecutionEnvelope` | `ContextAttentionExecutionEnvelope` | 去除 GQA 身份 |
| `bidirectional_gqa_attention` | `context_softmax_attention` | 名称表达 context+query 物理语义 |
| `bidirectional_gqa_attention_workspace_capacity_bytes` | `context_softmax_attention_workspace_capacity_bytes` | 与 entry 同名 |
| `include/ninfer/ops/swa.h` | `include/ninfer/ops/sliding_window_attention.h` | 删除缩写 header |
| `SwaContextExecutionEnvelope` | `SlidingWindowAttentionExecutionEnvelope` | 完整语义命名 |
| `swa` | `sliding_window_attention` | 保留当前窗口公式 |
| `swa_workspace_capacity_bytes` | `sliding_window_attention_workspace_capacity_bytes` | 与 entry 同名 |
| `include/ninfer/ops/kv_cache_append_prefix.h` | `include/ninfer/ops/kv_cache_append.h` | 与全量 append 组成 overload group |

项目自有接口不保留兼容性。最终不得存在旧函数 alias、旧类型 alias、转发 header、双注册 CMake
source 或同时维护的新旧测试。

### 6.2 源码文件

| 现有文件组 | 目标目录 |
|---|---|
| `src/ops/wrapper/gqa_attention.cpp` | `softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`；其中 standalone append 移入 `kv_cache/append/` |
| `src/ops/launcher/gqa_attention*` | `softmax_attention/dense/causal_cache/` |
| `src/ops/kernel/gqa_attention*` | Attention kernel 移入 `softmax_attention/dense/causal_cache/`；cache codec 移入 `kv_cache/codec.cuh`；全部符号去除 GQA |
| `src/ops/wrapper/vision_attention.cpp` | `softmax_attention/dense/packed/packed_softmax_attention.cpp` |
| `src/ops/launcher/vision_attention.*` | `softmax_attention/dense/packed/launch.*` |
| `src/ops/kernel/vision_attention.cuh` | `softmax_attention/dense/packed/kernel.cuh` |
| `src/ops/wrapper/bidirectional_gqa_attention.cpp` | `softmax_attention/dense/context/context_softmax_attention.cpp` |
| `src/ops/launcher/bidirectional_gqa_attention.*` | `softmax_attention/dense/context/launch.*` |
| `src/ops/wrapper/swa.cpp` | `softmax_attention/sliding_window/sliding_window_attention.cpp` |
| `src/ops/launcher/swa.*` | `softmax_attention/sliding_window/launch.*` |
| `src/ops/kernel/bidirectional_gqa_attention.cuh` | 拆为 `softmax_attention/common/context_query.cuh` 以及 context/sliding-window entry-local kernel |
| `src/ops/wrapper/kv_cache_append_prefix.cpp` 与对应 launcher/kernel | `kv_cache/append/`，并接收从 causal cache 分离出的 standalone append |

私有 C++/CUDA 类型同步按实际职责迁移：

| 现有私有名称 | 目标命名规则 |
|---|---|
| `GqaAttentionRoute` | `CausalAttentionRoute` |
| `GqaAppendInput` / `GqaCachedInput` | `CausalAppendInput` / `CausalCachedInput` |
| `Gqa27Geometry` / `Gqa35Geometry` | 以数值事实命名的 `CausalD256H24Kv4` / `CausalD256H16Kv2` |
| `BidirectionalGqaPlan` / `BidirectionalGqaRoute` | `ContextAttentionPlan` / `ContextAttentionRoute` |
| `SwaPlan` / `SwaRoute` | `SlidingWindowAttentionPlan` / `SlidingWindowAttentionRoute` |
| `VisionAttentionTile` | `PackedAttentionTile` |
| `kGqa*` / `kBidirectionalGqa*` / `kVisionAttention*` | 优先由已验证 geometry 导出；必须编译期固定时按 entry、format 或 tile 事实命名 |

数值 geometry 专用类型是私有 compile-time dispatch material，不是新的公共 Head 分类，也不
进入 target include。公共调用始终传 `AttentionHeadGeometry`。

`small_t`、`prompt`、`split_kv`、`bf16` 和 `i8` 可以继续作为私有 route/format 名称。`decode`
和 `prefill` 若只表示 token extent 对应的实现路线，应在迁移时优先改成 `small_t`、`prompt`
或具体算法名，避免把 Program phase 固化进实现身份。

### 6.3 调用方

| 调用方 | 迁移 |
|---|---|
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | include `softmax_attention.h`；Text/MTP 调用 `causal_softmax_attention` 或 `_cached`；standalone cache 写调用 `kv_cache_append` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h` | capacity 查询改用 geometry 和新 entry 名称 |
| `src/targets/qwen3_6/impl/runtime/vision_context_impl.h` | include `softmax_attention.h`；调用 `packed_softmax_attention`，不向 Op 传递 Vision 身份 |
| `src/targets/qwen3_6/impl/runtime/dflash_impl.h` | full route 调用 `context_softmax_attention`；local route 调用 `sliding_window_attention` |
| `src/targets/qwen3_6/impl/runtime/workspace_recipe.h` | target-local region 名可保留模型含义；Op capacity 和 shape 使用统一 geometry |

Family schedule 仍决定何时调用 full、local、Vision 或 MTP 路径；Op wrapper 不读取 layer role
或 target Variant。

### 6.4 Build ownership

`src/CMakeLists.txt` 继续显式列出每个 `.cpp`/`.cu`。迁移必须在一次目标态 cutover 中：

1. 添加新 source path；
2. 更新唯一 build owner；
3. 删除旧 horizontal source path；
4. 确认同一 kernel 或 launcher 没有被新旧 path 重复编译。

不改成 recursive glob，也不建立临时第二个 Attention library。

## 7. 测试与 benchmark 迁移

### 7.1 Qualification tests

Dense family 共享一个独立 FP64 oracle 和 Head mapping helper，但每个完整 entry 直接验证自己的
可见域、状态效果和注册实现路线。建议组织为：

```text
tests/ops/softmax_attention/
├── oracle.h
├── main.cpp
├── causal_cache.cpp
├── plain_and_packed.cpp
└── context.cpp

tests/ops/test_sliding_window_attention.cpp
tests/ops/test_kv_cache_append.cpp
```

对应测试 target：

```text
ninfer_softmax_attention_test
ninfer_sliding_window_attention_test
ninfer_kv_cache_append_test
```

现有测试迁移如下：

| 现有测试 | 目标 |
|---|---|
| `test_gqa_attention.cpp` | `softmax_attention/causal_cache.cpp`，codec-only 部分移入 `test_kv_cache_append.cpp` |
| `test_vision_attention.cpp` | `softmax_attention/plain_and_packed.cpp` |
| `test_bidirectional_gqa_attention.cpp` | `softmax_attention/context.cpp` |
| `test_swa.cpp` | `test_sliding_window_attention.cpp` |
| `test_kv_cache_append_prefix.cpp` | 合并进 `test_kv_cache_append.cpp` 的 prefix cases |

Oracle 必须从 geometry 和 entry 可见集合计算 Head 映射，不复制生产 kernel 的 group 常量。
必须覆盖：

- 四个当前注册 geometry；
- BF16 与 INT8-G64 causal cache profile；
- append-and-attend、cached-only 和 standalone append；
- plain single-segment entry，以及 packed 非等长与等长 segments；
- context 为零和非零时的 context+query 可见域；
- window 边界 `distance=window-1` 包含、`distance=window` 排除；
- CUDA Graph execution envelope 的有效 replay；
- cache code/scale bits、只读输入和全部声明的状态写入。

不同实现 profile 可以拥有不同命名 tolerance，但全部直接对同一个理想 Attention oracle
负责。

### 7.2 Benchmarks

长驻 benchmark 按公共语义分为：

| benchmark | 单次计时体的公共 entry |
|---|---|
| `ninfer_causal_softmax_attention_bench` | append-and-attend 或 cached-only |
| `ninfer_packed_softmax_attention_bench` | uniform plain segment 或 packed segments |
| `ninfer_context_softmax_attention_bench` | context-plus-query |
| `ninfer_sliding_window_attention_bench` | symmetric sliding window |
| `ninfer_kv_cache_append_bench` | full append 或 device-count prefix append |

fixture 构造、公共 capacity 查询、cache 条件化、CUDA Graph capture/instantiate 和同步都在计时
区间之外。eager 计时体恰好调用一次公共 Op；Graph 内也只 capture 同一次公共调用。
benchmark 不包含 private header、launcher、candidate、route forcing、tile/split 参数、生产
dispatch 复制或 kernel-name 正则。`--profile` 包围的是完整公共调用，调用内部出现的全部生产
kernel 都属于该 Op 的实测结果。

`attention_layer_bench` 不是闭合的公共 Op 契约，且原实现组合多个内部阶段并暴露 private
route，因此从长驻 Op benchmark 移除。完整 mixer 和 target 影响由公共 Engine 或 target round
benchmark 测量；private candidate 比较只允许存在于任务期临时代码，route 选定后删除。

## 8. 一次性迁移顺序

实现时按以下依赖顺序完成一个一致 cutover；这不是要求保留可发布的中间兼容态：

1. 在 `attention_geometry.h` 和 `softmax_attention.h` 中写出统一 geometry、公共公式、
   dense entries、envelopes 和 capacity contract。
2. 把 standalone `gqa_kv_append` 与现有 prefix append 收敛到 `kv_cache_append.h` 和
   `src/ops/kv_cache/append/`，确定唯一 cache codec 语义。
3. 将 causal cache、packed 和 context 实现移入 `softmax_attention/dense/`，先保持数值和
   route 选择不变，再消除 GQA/Vision/bidirectional 命名。
4. 将共享 context+query device body 提取为中性 common primitive，把 SWA 移入
   `sliding_window/` 并完整重命名。
5. 更新 target caller、workspace capacity、显式 CMake source、测试和 benchmark。
6. 更新受影响的 model reference、benchmark README 和测试命令。
7. 删除全部旧 header、source、symbol、target 和 forwarding path。
8. 运行 focused build、三类 Attention qualification、KV append qualification、相关 target
   执行和最终公共 benchmark。

若移动源码时发现现有 route 依赖旧 include 路径，应修正私有 ownership，不得以 forwarding
header 或复制 kernel 作为过渡终态。

## 9. 未来 Attention 的准入位置

未来名称先按真实语义归类：

| 常见名称 | 本规范中的归类 |
|---|---|
| MHA / MQA / GQA | `AttentionHeadGeometry`，不创建目录 |
| self / cross attention | 普通 `softmax_attention` 的 Q/K/V 来源和 `Tq/Tk` 取值 |
| causal / bidirectional / prefix mask | dense family 的明确可见域 entry；不预先增加任意 mask |
| packed / variable length | dense family 的输入表示 entry |
| local / sliding window | `sliding_window` 或在真实二维语义出现时创建独立 window family |
| global+local / block sparse / dilated sparse | 实际注册后创建 `structured_sparse/` peer family |
| MLA | 实际注册后创建 `multi_head_latent/` peer family；latent cache 与吸收投影不能展开成普通 K/V 再冒充复用 |
| deformable attention | 实际注册后创建独立 family，显式拥有 reference point、sampling 和聚合语义 |
| FlashAttention | dense/local family 内的私有 kernel/route |
| PagedAttention | cache storage/view 与私有寻址实现，不是数学 family |
| Ring Attention | 多设备执行策略；当前单 GPU 产品不准入 |

新增 family 必须由已注册目标和闭合 contract 驱动。只为了“扩展性”创建空目录、公共 enum、
runtime registry、通用 mask IR 或 backend interface 不属于扩展性，而是未被产品需要的框架。

### 9.1 术语参考

本规范使用下列原始工作区分数学语义、Head geometry 和实现策略：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：scaled dot-product、
  multi-head、self 和 encoder-decoder attention；
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
  与 [GQA](https://arxiv.org/abs/2305.13245)：MQA/GQA 的 KV Head 共享关系；
- [Longformer](https://arxiv.org/abs/2004.05150)、[BigBird](https://arxiv.org/abs/2007.14062)
  与 [Swin Transformer](https://arxiv.org/abs/2103.14030)：local、global+local、
  structured sparse 和二维窗口可见域；
- [FlashAttention](https://arxiv.org/abs/2205.14135) 与
  [PagedAttention](https://arxiv.org/abs/2309.06180)：IO-aware exact Attention 与 KV
  内存管理；
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)：MLA 的 latent KV cache 与吸收投影；
- [Deformable DETR](https://arxiv.org/abs/2010.04159)：reference-point sampling Attention。

## 10. 完成标准

迁移只有同时满足以下条件才完成：

- 目标源码树、共享 geometry header、三个公共 contract header 和唯一 CMake ownership
  已建立；
- 生产 target 只 include 新 semantic header；
- MHA/MQA/GQA 只作为 geometry 术语存在；
- Vision 身份不再出现在 Attention Op contract、源码 family、测试或 benchmark 名称中；
- SWA 公共名称完整迁移为 `sliding_window_attention`；
- standalone cache append 不再属于 Attention；
- 旧 header、旧 source、旧 symbol、旧测试 target、旧 benchmark target 和兼容 alias 已删除；
- dense、sliding-window 和 KV append 的独立 qualification 通过；
- 两个注册目标的 Text、Vision、MTP、DFlash、prefix reuse 和 CUDA Graph 路径保持支持；
- 受影响文档和命令使用新名称，除本文的迁移映射外没有陈旧引用；
- 相关公共 benchmark 证明迁移未造成目标路径的性能回退，或对有意的性能变化给出直接证据。
