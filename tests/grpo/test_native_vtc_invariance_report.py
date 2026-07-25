import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF_SCRIPT = REPO_ROOT / "scripts/prove_fugu_native_grpo_vtc_invariance_v1.py"
PROOF_REPORT = REPO_ROOT / "tests/grpo/artifacts/fugu_native_grpo_vtc_invariance_v1.json"
NATIVE_SOURCE = REPO_ROOT / "csrc/src/runtime/dsl/dsl_model_execution.cpp"
EXPECTED_REPORT_SHA256 = "34f008c4593ab343b1fdebe6a0fb404c773cc9e9eec980db98b6451d3ee7cb2d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_zero_paid_native_final_vtc_invariance_proof_is_current_and_passes() -> None:
    report = json.loads(PROOF_REPORT.read_text())

    assert _sha256(PROOF_REPORT) == EXPECTED_REPORT_SHA256
    assert report["schema_version"] == "fugu.native_grpo.final_vtc_invariance.v1"
    assert report["verdict"] == "PASS"
    assert report["scope"] == {
        "campaign_27b_checkpoint_touched": False,
        "campaign_optimizer_steps": 0,
        "local_small_model_optimizer_updates": 2,
        "paid_calls": 0,
        "paid_collection_artifacts_touched": False,
        "proof_model_only": True,
    }
    assert report["proof_source"]["sha256"] == _sha256(PROOF_SCRIPT)
    assert report["runtime"]["native_source"]["sha256"] == _sha256(NATIVE_SOURCE)
    assert report["runtime"]["imported_extension_matches_rebuilt"] is True
    assert report["runtime"]["loaded_common_is_rebuilt"] is True
    assert report["runtime"]["native_token_scaling_fix_present"] is True
    assert report["model"]["name"] == "Qwen/Qwen3.5-0.8B"
    assert report["model"]["network_downloads"] == 0

    variant_1 = report["variants"]["final_vtc_1"]
    variant_8 = report["variants"]["final_vtc_8"]
    assert variant_1["requested_final_vtc"] == variant_1["observed_final_vtc"] == 1
    assert variant_8["requested_final_vtc"] == variant_8["observed_final_vtc"] == 8
    assert (
        variant_1["initial_weights"]["digest_sha256"]
        == variant_8["initial_weights"]["digest_sha256"]
    )
    assert (
        variant_1["accumulated_gradients"]["digest_sha256"]
        == variant_8["accumulated_gradients"]["digest_sha256"]
    )
    assert variant_1["optimizer_norm"] == variant_8["optimizer_norm"]
    assert (
        variant_1["post_update_weights"]["digest_sha256"]
        == variant_8["post_update_weights"]["digest_sha256"]
    )
    assert set(report["assertions"].values()) == {True}
    assert set(report["comparisons"].values()) == {0.0}
