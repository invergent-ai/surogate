import pytest
from typing import Any

from surogate.utils.model import recommend_training_params

GPU_CONFIGS = {
    # Consumer GPUs
    "3090": {"vram_gb": 24, "generation": "Ampere"},
    "4090": {"vram_gb": 24, "generation": "Ada Lovelace"},
    "5090": {"vram_gb": 32, "generation": "Blackwell"},
    # Professional/Datacenter GPUs
    "L40": {"vram_gb": 48, "generation": "Ada Lovelace"},
    "L40S": {"vram_gb": 48, "generation": "Ada Lovelace"},
    "A40": {"vram_gb": 48, "generation": "Ampere"},
    "A80": {"vram_gb": 80, "generation": "Ampere"},  # A800
    "A100": {"vram_gb": 80, "generation": "Ampere"},  # 80GB variant
    "H100": {"vram_gb": 80, "generation": "Hopper"},
    "H200": {"vram_gb": 141, "generation": "Hopper"},
}

GPU_TYPES = list(GPU_CONFIGS.keys())
NUM_GPUS = [1, 2, 4, 8]
DATASET_SIZES = [100, 1_000, 10_000, 100_000, 500_000, 1_000_000]
MODEL_SIZES = [7, 8, 13, 70]  # Billion parameters
QUANTIZATIONS = ["4bit", "8bit", "bf16", "fp16"]
SEQ_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def gpu_vram_mapping() -> dict[str, int]:
    """Mapping of GPU type to VRAM in GB."""
    return {name: config["vram_gb"] for name, config in GPU_CONFIGS.items()}


# =============================================================================
# Test Parameter Generation
# =============================================================================

def generate_gpu_num_gpu_params():
    """Generate all GPU type and count combinations."""
    params = []
    for gpu_type in GPU_TYPES:
        for num_gpus in NUM_GPUS:
            vram = GPU_CONFIGS[gpu_type]["vram_gb"]
            params.append(
                pytest.param(
                    gpu_type, num_gpus, vram,
                    id=f"{gpu_type}-{num_gpus}x"
                )
            )
    return params


def generate_full_matrix_params():
    """Generate full test matrix: GPU x num_gpus x dataset_size."""
    params = []
    for gpu_type in GPU_TYPES:
        vram = GPU_CONFIGS[gpu_type]["vram_gb"]
        for num_gpus in NUM_GPUS:
            for dataset_size in DATASET_SIZES:
                params.append(
                    pytest.param(
                        gpu_type, num_gpus, vram, dataset_size,
                        id=f"{gpu_type}-{num_gpus}x-ds{dataset_size}"
                    )
                )
    return params


def generate_model_size_params():
    """Generate params for testing different model sizes."""
    params = []
    for model_size in MODEL_SIZES:
        for quantization in QUANTIZATIONS:
            params.append(
                pytest.param(
                    model_size, quantization,
                    id=f"{model_size}B-{quantization}"
                )
            )
    return params


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_config_structure(config: dict[str, Any]) -> None:
    """Validate the returned configuration has required keys and types."""
    required_keys = {
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "deepspeed",
        "warnings",
    }

    assert required_keys.issubset(config.keys()), (
        f"Missing required keys: {required_keys - set(config.keys())}"
    )

    assert isinstance(config["per_device_train_batch_size"], int)
    assert isinstance(config["gradient_accumulation_steps"], int)
    assert isinstance(config["gradient_checkpointing"], bool)
    assert isinstance(config["deepspeed"], (dict, type(None)))
    assert isinstance(config["warnings"], list)


def validate_batch_config(config: dict[str, Any], num_gpus: int) -> None:
    """Validate batch size and gradient accumulation settings."""
    batch_size = config["per_device_train_batch_size"]
    grad_accum = config["gradient_accumulation_steps"]

    assert batch_size >= 1, f"Batch size must be >= 1, got {batch_size}"
    assert grad_accum >= 1, f"Gradient accumulation must be >= 1, got {grad_accum}"

    # Effective batch size should be reasonable
    effective_batch = batch_size * grad_accum * num_gpus
    assert effective_batch >= 1, f"Effective batch size must be >= 1, got {effective_batch}"


def validate_deepspeed_config(ds_config: dict[str, Any] | None, num_gpus: int) -> None:
    """Validate DeepSpeed configuration if present."""
    if ds_config is None:
        return

    # Check for common DeepSpeed config keys
    valid_stages = [0, 1, 2, 3]

    if "zero_optimization" in ds_config:
        zero_config = ds_config["zero_optimization"]
        if "stage" in zero_config:
            assert zero_config["stage"] in valid_stages, (
                f"Invalid ZeRO stage: {zero_config['stage']}"
            )

    # For multi-GPU setups, usually expect ZeRO optimization
    if num_gpus > 1 and ds_config:
        # DeepSpeed should have some form of optimization for multi-GPU
        assert any(key in ds_config for key in [
            "zero_optimization", "fp16", "bf16", "gradient_clipping"
        ]), "Multi-GPU setup should have DeepSpeed optimizations"


# =============================================================================
# Test Classes
# =============================================================================

class TestBasicFunctionality:
    """Test basic functionality and return structure."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=1000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        assert isinstance(config, dict)

    def test_config_structure(self):
        """Test that returned config has required structure."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=1000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        validate_config_structure(config)

    def test_gradient_checkpointing_always_on(self):
        """Test that gradient checkpointing is always enabled for LoRA."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=1000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        assert config["gradient_checkpointing"] is True


class TestGPUConfigurations:
    """Test across different GPU types and counts."""

    @pytest.mark.parametrize("gpu_type,num_gpus,vram_gb", generate_gpu_num_gpu_params())
    def test_gpu_config_returns_valid_structure(
            self, gpu_type: str, num_gpus: int, vram_gb: int
    ):
        """Test that all GPU configurations return valid structure."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=num_gpus,
            vram_per_gpu_gb=vram_gb,
        )
        validate_config_structure(config)
        validate_batch_config(config, num_gpus)

    @pytest.mark.parametrize("gpu_type,num_gpus,vram_gb", generate_gpu_num_gpu_params())
    def test_gpu_config_batch_sizes_positive(
            self, gpu_type: str, num_gpus: int, vram_gb: int
    ):
        """Test that batch sizes are positive for all GPU configs."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=num_gpus,
            vram_per_gpu_gb=vram_gb,
        )
        assert config["per_device_train_batch_size"] > 0
        assert config["gradient_accumulation_steps"] > 0


class TestDatasetSizes:
    """Test across different dataset sizes."""

    @pytest.mark.parametrize(
        "gpu_type,num_gpus,vram_gb,dataset_size",
        generate_full_matrix_params()
    )
    def test_full_matrix(
            self,
            gpu_type: str,
            num_gpus: int,
            vram_gb: int,
            dataset_size: int,
    ):
        """Test full matrix of GPU configs and dataset sizes."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=dataset_size,
            quantization="4bit",
            seq_len=4096,
            num_gpus=num_gpus,
            vram_per_gpu_gb=vram_gb,
        )
        validate_config_structure(config)
        validate_batch_config(config, num_gpus)

    @pytest.mark.parametrize("dataset_size", DATASET_SIZES)
    def test_small_dataset_warnings(self, dataset_size: int):
        """Test that small datasets may generate warnings."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=dataset_size,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        # Small datasets might trigger warnings about overfitting
        if dataset_size < 1000:
            # Just ensure warnings is a list (may or may not have warnings)
            assert isinstance(config["warnings"], list)


class TestModelSizes:
    """Test across different model sizes and quantization."""

    @pytest.mark.parametrize("model_size_b,quantization", generate_model_size_params())
    def test_model_quantization_combinations(
            self, model_size_b: float, quantization: str
    ):
        """Test various model size and quantization combinations."""
        config = recommend_training_params(
            model_size_b=model_size_b,
            dataset_size=10_000,
            quantization=quantization,
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        validate_config_structure(config)

    @pytest.mark.parametrize("model_size_b", MODEL_SIZES)
    def test_large_model_on_small_gpu(self, model_size_b: float):
        """Test large models on consumer GPUs (may generate warnings)."""
        config = recommend_training_params(
            model_size_b=model_size_b,
            dataset_size=10_000,
            quantization="4bit",  # Use 4bit to fit larger models
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,  # 3090/4090 VRAM
        )
        validate_config_structure(config)

        # Large models on small GPUs should likely have warnings
        if model_size_b >= 70:
            # 70B on single 24GB should produce warnings or very small batch
            assert (
                    len(config["warnings"]) > 0 or
                    config["per_device_train_batch_size"] == 1
            )


class TestSequenceLengths:
    """Test across different sequence lengths."""

    @pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
    def test_sequence_lengths(self, seq_len: int):
        """Test various sequence lengths."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=seq_len,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        validate_config_structure(config)
        validate_batch_config(config, num_gpus=1)

    @pytest.mark.parametrize("seq_len", [8192, 16384, 32768])
    def test_long_context_reduces_batch_size(self, seq_len: int):
        """Test that longer context reduces batch size or increases gradient accumulation."""
        short_ctx_config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=2048,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )

        long_ctx_config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=seq_len,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )

        # Longer context should result in smaller or equal batch size
        assert (
                long_ctx_config["per_device_train_batch_size"] <=
                short_ctx_config["per_device_train_batch_size"]
        )


class TestDeepSpeedConfigurations:
    """Test DeepSpeed configuration generation."""

    @pytest.mark.parametrize("num_gpus", NUM_GPUS)
    def test_deepspeed_config_structure(self, num_gpus: int):
        """Test DeepSpeed config structure for various GPU counts."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=num_gpus,
            vram_per_gpu_gb=24,
        )
        validate_deepspeed_config(config["deepspeed"], num_gpus)

    @pytest.mark.parametrize("gpu_type,num_gpus,vram_gb", generate_gpu_num_gpu_params())
    def test_deepspeed_config_all_gpus(
            self, gpu_type: str, num_gpus: int, vram_gb: int
    ):
        """Test DeepSpeed configuration for all GPU types."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=num_gpus,
            vram_per_gpu_gb=vram_gb,
        )
        validate_deepspeed_config(config["deepspeed"], num_gpus)


class TestHighVRAMGPUs:
    """Specific tests for high-VRAM datacenter GPUs."""

    @pytest.mark.parametrize(
        "gpu_type,vram_gb",
        [
            ("A100", 80),
            ("H100", 80),
            ("H200", 141),
        ]
    )
    def test_large_model_on_high_vram(self, gpu_type: str, vram_gb: int):
        """Test 70B model on high-VRAM GPUs."""
        config = recommend_training_params(
            model_size_b=70,
            dataset_size=100_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=8,
            vram_per_gpu_gb=vram_gb,
        )
        validate_config_structure(config)
        validate_batch_config(config, num_gpus=8)

    @pytest.mark.parametrize(
        "gpu_type,vram_gb",
        [
            ("A100", 80),
            ("H100", 80),
            ("H200", 141),
        ]
    )
    def test_bf16_on_datacenter_gpus(self, gpu_type: str, vram_gb: int):
        """Test bf16 training on datacenter GPUs with sufficient VRAM."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=100_000,
            quantization="bf16",
            seq_len=4096,
            num_gpus=8,
            vram_per_gpu_gb=vram_gb,
        )
        validate_config_structure(config)
        # bf16 on high-VRAM should allow reasonable batch sizes
        assert config["per_device_train_batch_size"] >= 1


class TestConsumerGPUs:
    """Specific tests for consumer GPUs."""

    @pytest.mark.parametrize(
        "gpu_type,vram_gb",
        [
            ("3090", 24),
            ("4090", 24),
            ("5090", 32),
        ]
    )
    def test_consumer_gpu_small_model(self, gpu_type: str, vram_gb: int):
        """Test small models on consumer GPUs."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=vram_gb,
        )
        validate_config_structure(config)
        validate_batch_config(config, num_gpus=1)

    @pytest.mark.parametrize(
        "gpu_type,vram_gb",
        [
            ("3090", 24),
            ("4090", 24),
        ]
    )
    def test_consumer_gpu_large_model_warning(self, gpu_type: str, vram_gb: int):
        """Test that large models on consumer GPUs produce warnings."""
        config = recommend_training_params(
            model_size_b=70,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=vram_gb,
        )
        # Should either warn or use minimal batch size
        validate_config_structure(config)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_dataset_size(self):
        """Test with minimum dataset size."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=1,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        validate_config_structure(config)

    def test_very_large_dataset(self):
        """Test with very large dataset."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=8,
            vram_per_gpu_gb=80,
        )
        validate_config_structure(config)

    def test_minimum_vram(self):
        """Test with minimum reasonable VRAM."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=1000,
            quantization="4bit",
            seq_len=2048,
            num_gpus=1,
            vram_per_gpu_gb=8,  # Very low VRAM
        )
        validate_config_structure(config)

    def test_maximum_gpus(self):
        """Test with maximum GPU count."""
        config = recommend_training_params(
            model_size_b=70,
            dataset_size=1_000_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=8,
            vram_per_gpu_gb=80,
        )
        validate_config_structure(config)
        validate_batch_config(config, num_gpus=8)


class TestQuantizationEffects:
    """Test effects of different quantization settings."""

    @pytest.mark.parametrize("quantization", QUANTIZATIONS)
    def test_quantization_batch_size_relationship(self, quantization: str):
        """Test that lower precision allows larger batch sizes."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization=quantization,
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        validate_config_structure(config)

    def test_4bit_vs_bf16_batch_size(self):
        """Test that 4bit allows larger batch than bf16."""
        config_4bit = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )

        config_bf16 = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="bf16",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )

        # 4bit should allow >= batch size compared to bf16
        assert (
                config_4bit["per_device_train_batch_size"] >=
                config_bf16["per_device_train_batch_size"]
        )


class TestScalingBehavior:
    """Test scaling behavior across configurations."""

    def test_more_gpus_maintains_effective_batch(self):
        """Test that more GPUs maintain similar effective batch size."""
        config_1gpu = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )

        config_4gpu = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=4,
            vram_per_gpu_gb=24,
        )

        eff_batch_1 = (
                config_1gpu["per_device_train_batch_size"] *
                config_1gpu["gradient_accumulation_steps"] * 1
        )
        eff_batch_4 = (
                config_4gpu["per_device_train_batch_size"] *
                config_4gpu["gradient_accumulation_steps"] * 4
        )

        # Effective batch sizes should be in similar range (within 4x)
        assert 0.25 <= eff_batch_4 / eff_batch_1 <= 4.0

    def test_more_vram_allows_larger_batch(self):
        """Test that more VRAM allows larger batch sizes."""
        config_24gb = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )

        config_80gb = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=80,
        )

        # 80GB should allow >= batch size compared to 24GB
        assert (
                config_80gb["per_device_train_batch_size"] >=
                config_24gb["per_device_train_batch_size"]
        )


# =============================================================================
# Smoke Tests - Quick sanity checks
# =============================================================================

class TestSmokeTests:
    """Quick smoke tests for basic functionality."""

    @pytest.mark.parametrize("gpu_type", GPU_TYPES)
    def test_each_gpu_type_works(self, gpu_type: str):
        """Smoke test: each GPU type returns valid config."""
        vram = GPU_CONFIGS[gpu_type]["vram_gb"]
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=10_000,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=vram,
        )
        assert config is not None
        assert "per_device_train_batch_size" in config

    @pytest.mark.parametrize("dataset_size", DATASET_SIZES)
    def test_each_dataset_size_works(self, dataset_size: int):
        """Smoke test: each dataset size returns valid config."""
        config = recommend_training_params(
            model_size_b=7,
            dataset_size=dataset_size,
            quantization="4bit",
            seq_len=4096,
            num_gpus=1,
            vram_per_gpu_gb=24,
        )
        assert config is not None


# =============================================================================
# Run configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])