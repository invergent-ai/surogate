from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DSL_EXECUTION = REPO_ROOT / "csrc/src/runtime/dsl/dsl_model_execution.cpp"
FUSED_CLASSIFIER = REPO_ROOT / "csrc/src/kernels/fused_classifier.cu"
GLOBAL_NORM = REPO_ROOT / "csrc/src/kernels/global_norm.cu"
BINDING = REPO_ROOT / "csrc/src/binding/binding.cpp"


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for idx in range(brace, len(source)):
        char = source[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace : idx + 1]
    raise AssertionError(f"could not parse body for {signature}")


def test_legacy_grpo_and_custom_loss_do_not_allocate_transient_gpu_buffers():
    source = DSL_EXECUTION.read_text()
    no_alloc_signatures = [
        "std::vector<float> DslModel::compute_logprobs(",
        "void DslModel::step_with_custom_loss(",
        "std::vector<float> DslModel::forward_for_grpo(",
        "void DslModel::backward_grpo(",
        "void DslModel::step_grpo_native(",
    ]

    for signature in no_alloc_signatures:
        body = _function_body(source, signature)
        assert "cudaMalloc(" not in body, signature
        assert "cudaFree(" not in body, signature

    no_sync_signatures = [
        "void DslModel::step_with_custom_loss(",
        "void DslModel::backward_grpo(",
        "void DslModel::step_grpo_native(",
    ]
    for signature in no_sync_signatures:
        body = _function_body(source, signature)
        assert "cudaStreamSynchronize(" not in body, signature


def test_native_replay_is_ce_only_before_behavior_ratio_or_kl() -> None:
    source = FUSED_CLASSIFIER.read_text()
    replay_branch = source.index("const bool is_replay =")
    ratio = source.index("const float log_importance_ratio =", replay_branch)
    replay_continue = source.index("continue;", replay_branch)

    assert replay_branch < replay_continue < ratio
    branch = source[replay_branch:replay_continue]
    assert "custom_dloss[out_idx] = replay_tau * replay_weight * inv_loss_scale" in branch
    assert "const float replay_weight = replay_weights ? replay_weights[logical_idx] : 1.0f" in branch
    assert "kl_tau" not in branch
    assert "advantages[logical_idx]" not in branch


def test_all_explicit_grpo_loss_paths_disable_legacy_optimizer_token_scaling() -> None:
    source = DSL_EXECUTION.read_text()
    explicit_loss_signatures = [
        "void DslModel::step_with_custom_loss(",
        "std::vector<float> DslModel::forward_for_grpo(",
        "void DslModel::step_grpo_native(",
    ]

    for signature in explicit_loss_signatures:
        body = _function_body(source, signature)
        assert "mUseTokenScale = false;" in body, signature


def test_native_lora_gradient_preflight_is_read_only() -> None:
    source = DSL_EXECUTION.read_text()
    body = _function_body(
        source,
        "float DslModel::preflight_grpo_native_lora_gradient_norm(",
    )

    assert "calculate_lora_gradient_norm(comm, grad_clip);" in body
    assert "update_lora_" not in body
    assert "update_with_config" not in body
    assert "advance_sync_generation" not in body


def test_lora_multi_tensor_amax_propagates_nonfinite_gradient_sentinel() -> None:
    source = GLOBAL_NORM.read_text()
    body = _function_body(source, "__global__ void global_amax_multi_tensor_kernel(")

    assert body.count("if (!isfinite(raw))") == 2
    assert body.count("atomicMax(reinterpret_cast<unsigned int*>(amax_out), 0x7fc00000u);") == 2


def test_native_mismatch_metrics_exclude_replay_only_samples() -> None:
    source = FUSED_CLASSIFIER.read_text()
    body = _function_body(source, "__global__ void grpo_custom_dloss_kernel(")

    replay_branch = body.index("if (is_replay)")
    ratio_branch = body.index("const float inference_logprob", replay_branch)
    policy_count = body.index("policy_count += 1.0f;", ratio_branch)
    policy_guard = body.index("if (sample_policy_count > 0.0f)")
    mismatch_add = body.index("atomicAdd(metrics + GRPO_METRIC_MISMATCH_KL", policy_guard)
    replay_continue = body.index("continue;", replay_branch)

    assert replay_continue < ratio_branch < policy_count < policy_guard < mismatch_add
    assert "atomicAdd(metrics + GRPO_METRIC_POLICY_SAMPLE_COUNT, 1.0f);" in body


def test_forward_for_grpo_shapes_only_the_native_returned_buffer() -> None:
    source = BINDING.read_text()
    body = _function_body(source, '"forward_for_grpo",')

    assert "n == 0 || T == 0 || n % T != 0" in body
    assert "const std::size_t output_rows = n / T;" in body
    assert "{output_rows, T}" in body
    assert "{B, T}" not in body
