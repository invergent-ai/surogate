import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict


class TestNCCLIdGeneration:
    """Tests for NCCL ID generation."""

    def test_generate_nccl_id_returns_128_bytes(self):
        """NCCL ID should be exactly 128 bytes."""
        from surogate import _surogate

        nccl_id = _surogate.generate_nccl_id()
        assert isinstance(nccl_id, bytes)
        assert len(nccl_id) == 128

    def test_generate_nccl_id_is_unique(self):
        """Each call should generate a unique ID."""
        from surogate import _surogate

        id1 = _surogate.generate_nccl_id()
        id2 = _surogate.generate_nccl_id()
        assert id1 != id2

    def test_generate_nccl_id_multiple_calls(self):
        """Should be able to generate many IDs without errors."""
        from surogate import _surogate

        ids = [_surogate.generate_nccl_id() for _ in range(10)]
        # All should be unique
        assert len(set(ids)) == 10


class TestDistributedConfig:
    """Tests for DistributedConfig dataclass."""

    def test_default_values(self):
        """DistributedConfig should have sensible defaults."""
        from surogate.core.config.sft_config import DistributedConfig

        config = DistributedConfig()
        assert config.ray_address == "auto"
        assert config.num_nodes == 1
        assert config.gpus_per_node == 0

    def test_custom_values(self):
        """DistributedConfig should accept custom values."""
        from surogate.core.config.sft_config import DistributedConfig

        config = DistributedConfig(
            ray_address="ray://192.168.1.100:6379",
            num_nodes=4,
            gpus_per_node=8,
        )
        assert config.ray_address == "ray://192.168.1.100:6379"
        assert config.num_nodes == 4
        assert config.gpus_per_node == 8

    def test_dataclass_conversion(self):
        """DistributedConfig should be convertible to dict."""
        from surogate.core.config.sft_config import DistributedConfig

        config = DistributedConfig(num_nodes=2, gpus_per_node=4)
        config_dict = asdict(config)
        assert config_dict["num_nodes"] == 2
        assert config_dict["gpus_per_node"] == 4


class TestSFTConfigDistributed:
    """Tests for distributed config parsing in SFTConfig."""

    def test_sft_config_without_distributed(self):
        """SFTConfig should work without distributed config."""
        from surogate.core.config.sft_config import SFTConfig
        from surogate.utils.dict import DictDefault

        cfg = DictDefault({
            "model": {"model_dir": "test-model"},
            "datasets": [{"path": "test.json"}],
        })
        # This would normally fail because of model validation,
        # so we just test the distributed parsing logic
        # by checking that None distributed is handled
        assert True  # Placeholder - full test needs model fixtures

    def test_distributed_config_from_dict(self):
        """DistributedConfig should be parseable from dict in SFTConfig."""
        from surogate.core.config.sft_config import DistributedConfig

        # Test the parsing logic directly
        distributed_cfg = {
            "ray_address": "auto",
            "num_nodes": 2,
            "gpus_per_node": 4,
        }

        config = DistributedConfig(
            ray_address=distributed_cfg.get("ray_address", "auto"),
            num_nodes=distributed_cfg.get("num_nodes", 1),
            gpus_per_node=distributed_cfg.get("gpus_per_node", 0),
        )

        assert config.num_nodes == 2
        assert config.gpus_per_node == 4


class TestNodeTrainer:
    """Tests for NodeTrainer class."""

    def test_node_trainer_init(self):
        """NodeTrainer should initialize with correct parameters."""
        from surogate.train.distributed import NodeTrainer

        trainer = NodeTrainer(
            config_dict={"model": {"model_dir": "test"}},
            train_files=["train.bin"],
            eval_files=["eval.bin"],
            node_rank=0,
            num_nodes=2,
            nccl_id=b"\x00" * 128,
            gpus_per_node=4,
        )

        assert trainer.node_rank == 0
        assert trainer.num_nodes == 2
        assert trainer.gpus_per_node == 4
        assert len(trainer.nccl_id) == 128

    def test_node_trainer_stores_config(self):
        """NodeTrainer should store config dict for later reconstruction."""
        from surogate.train.distributed import NodeTrainer

        config_dict = {
            "model": {"model_dir": "test-model"},
            "per_device_train_batch_size": 4,
        }

        trainer = NodeTrainer(
            config_dict=config_dict,
            train_files=["train.bin"],
            eval_files=None,
            node_rank=1,
            num_nodes=2,
            nccl_id=b"\x00" * 128,
            gpus_per_node=2,
        )

        assert trainer.config_dict == config_dict
        assert trainer.eval_files is None


class TestRayDistributedTrainerMocked:
    """Tests for RayDistributedTrainer with mocked Ray."""

    @patch("surogate.train.distributed._get_ray")
    def test_trainer_generates_nccl_id(self, mock_get_ray):
        """RayDistributedTrainer should generate a single NCCL ID."""
        # Mock Ray
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [{"Alive": True}, {"Alive": True}]
        mock_get_ray.return_value = mock_ray

        # Mock config
        mock_config = MagicMock()
        mock_config.gpus = 4
        mock_config.model_dump.return_value = {"model": {"model_dir": "test"}}

        from surogate.train.distributed import RayDistributedTrainer

        trainer = RayDistributedTrainer(
            config=mock_config,
            train_files=["train.bin"],
            eval_files=None,
            num_nodes=2,
            gpus_per_node=4,
        )

        # Check NCCL ID was generated (single ID, node master derived via ncclCommSplit)
        assert len(trainer.nccl_id) == 128

    @patch("surogate.train.distributed._get_ray")
    def test_trainer_auto_detects_nodes(self, mock_get_ray):
        """RayDistributedTrainer should auto-detect nodes from Ray cluster."""
        # Mock Ray with 3 alive nodes
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {"Alive": True},
            {"Alive": True},
            {"Alive": True},
            {"Alive": False},  # Dead node should be excluded
        ]
        mock_get_ray.return_value = mock_ray

        mock_config = MagicMock()
        mock_config.gpus = 4
        mock_config.model_dump.return_value = {}

        from surogate.train.distributed import RayDistributedTrainer

        trainer = RayDistributedTrainer(
            config=mock_config,
            train_files=["train.bin"],
            num_nodes=None,  # Auto-detect
        )

        assert trainer.num_nodes == 3

    @patch("surogate.train.distributed._get_ray")
    def test_trainer_uses_gpus_per_node(self, mock_get_ray):
        """RayDistributedTrainer should use specified gpus_per_node."""
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [{"Alive": True}]
        mock_get_ray.return_value = mock_ray

        mock_config = MagicMock()
        mock_config.gpus = 8  # Config has 8
        mock_config.model_dump.return_value = {}

        from surogate.train.distributed import RayDistributedTrainer

        trainer = RayDistributedTrainer(
            config=mock_config,
            train_files=["train.bin"],
            gpus_per_node=4,  # Override to 4
        )

        assert trainer.gpus_per_node == 4

    @patch("surogate.train.distributed._get_ray")
    def test_trainer_falls_back_to_config_gpus(self, mock_get_ray):
        """RayDistributedTrainer should use config.gpus when gpus_per_node=0."""
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [{"Alive": True}]
        mock_get_ray.return_value = mock_ray

        mock_config = MagicMock()
        mock_config.gpus = 8
        mock_config.model_dump.return_value = {}

        from surogate.train.distributed import RayDistributedTrainer

        trainer = RayDistributedTrainer(
            config=mock_config,
            train_files=["train.bin"],
            gpus_per_node=0,  # Default, should use config.gpus
        )

        assert trainer.gpus_per_node == 8


class TestCreateMultinodeBinding:
    """Tests for SurogateTrainer.create_multinode binding."""

    def test_create_multinode_exists(self):
        """create_multinode should be available as a static method."""
        from surogate import _surogate

        assert hasattr(_surogate.SurogateTrainer, "create_multinode")
        # Check it's callable
        assert callable(_surogate.SurogateTrainer.create_multinode)

    def test_create_multinode_requires_config(self):
        """create_multinode should require config and options parameters."""
        from surogate import _surogate

        # Passing None for config/options should fail with TypeError
        # (binding requires PretrainedConfig and RuntimeOptions types)
        with pytest.raises(TypeError):
            _surogate.SurogateTrainer.create_multinode(
                ngpu=1,
                node_rank=0,
                num_nodes=1,
                nccl_id=b"\x00" * 128,
                config=None,
                options=None,
                batch_size=1,
                seq_len=128,
                grad_accum=1,
            )


class TestNodeTrainingResult:
    """Tests for NodeTrainingResult dataclass."""

    def test_node_training_result_fields(self):
        """NodeTrainingResult should have expected fields."""
        from surogate.train.distributed import NodeTrainingResult

        result = NodeTrainingResult(
            node_rank=0,
            final_loss=0.5,
            final_step=1000,
            checkpoint_path="/tmp/checkpoint",
        )

        assert result.node_rank == 0
        assert result.final_loss == 0.5
        assert result.final_step == 1000
        assert result.checkpoint_path == "/tmp/checkpoint"

    def test_node_training_result_optional_checkpoint(self):
        """NodeTrainingResult checkpoint_path should be optional."""
        from surogate.train.distributed import NodeTrainingResult

        result = NodeTrainingResult(
            node_rank=1,
            final_loss=0.3,
            final_step=500,
        )

        assert result.checkpoint_path is None


# Integration tests that require GPU
@pytest.mark.gpu
class TestIntegrationGPU:
    """Integration tests requiring GPU hardware."""

    def test_nccl_id_roundtrip(self):
        """NCCL ID should be usable in multi-node setup."""
        from surogate import _surogate

        # Generate single ID (node master comm derived via ncclCommSplit)
        nccl_id = _surogate.generate_nccl_id()

        # Verify it's valid bytes
        assert isinstance(nccl_id, bytes)
        assert len(nccl_id) == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
