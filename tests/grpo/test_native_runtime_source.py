from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DSL_EXECUTION = REPO_ROOT / "csrc/src/runtime/dsl/dsl_model_execution.cpp"


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
