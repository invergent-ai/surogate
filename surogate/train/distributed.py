from __future__ import annotations

from copy import deepcopy
import glob
import os
import shutil
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from surogate.core.model.loader import get_model_info_and_tokenizer

if TYPE_CHECKING:
    from surogate.core.config.sft_config import SFTConfig

# Lazy import Ray to avoid dependency when not using distributed training
_ray = None


def _get_ray():
    """Lazy import Ray."""
    global _ray
    if _ray is None:
        try:
            import ray
            _ray = ray
        except ImportError:
            raise ImportError(
                "Ray is required for distributed training. "
                "Install with: pip install surogate[distributed]"
            )
    return _ray


def _serialize_config_value(value: Any) -> Any:
    """
    Recursively serialize config values to be Ray-compatible.

    Converts Path objects to strings and dataclass objects to dicts.
    Skips C++ objects that can't be serialized (they'll be reconstructed on workers).
    """
    # Skip C++ objects
    if hasattr(value, '__module__') and 'surogate._surogate' in str(value.__module__):
        return None

    # Convert Path to string
    if isinstance(value, Path):
        return str(value)

    # Convert dataclass to dict
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _serialize_config_value(v) for k, v in value.__dict__.items()}

    # Recursively handle lists
    if isinstance(value, list):
        return [_serialize_config_value(v) for v in value]

    # Recursively handle dicts
    if isinstance(value, dict):
        return {k: _serialize_config_value(v) for k, v in value.items()}

    # Return primitives as-is
    return value


@dataclass
class NodeTrainingResult:
    """Result from a single node's training."""
    node_rank: int
    final_loss: float
    final_step: int
    checkpoint_path: Optional[str] = None


def _create_barrier_actor(num_nodes: int):
    """
    Create a Ray actor that implements a barrier for synchronizing nodes.

    This is needed because NCCL initialization requires all nodes to call
    ncclCommInitRank at nearly the same time. Ray's ray.get() only ensures
    tasks complete, not that they start simultaneously.
    """
    ray = _get_ray()

    @ray.remote
    class BarrierActor:
        def __init__(self, num_participants: int):
            self.num_participants = num_participants
            self.arrived = 0
            self.generation = 0

        def arrive(self) -> Tuple[int, int]:
            """
            Register arrival at barrier. Returns (generation, arrived_count).
            Non-blocking - caller must poll check_ready() separately.
            """
            current_gen = self.generation
            self.arrived += 1
            count = self.arrived

            # If we're the last one, reset for next use
            if self.arrived >= self.num_participants:
                self.arrived = 0
                self.generation += 1

            return (current_gen, count)

        def get_generation(self) -> int:
            """Check current generation (for polling)."""
            return self.generation

    return BarrierActor.remote(num_nodes)


class NodeTrainer:
    """
    Training worker that runs on a single node.

    Uses the existing single-node threaded backend (MultiGPUPyTrainer) for
    local GPU communication, and coordinates with other nodes via NCCL.
    """

    def __init__(
        self,
        config_dict: Dict[str, Any],
        train_files: List[str],
        eval_files: Optional[List[str]],
        node_rank: int,
        num_nodes: int,
        nccl_id: bytes,
        node_master_nccl_id: bytes,
        gpus_per_node: int,
        tokenize_on_node: bool = False,
    ):
        self.config_dict = config_dict
        self.train_files = train_files
        self.eval_files = eval_files
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.nccl_id = nccl_id
        self.node_master_nccl_id = node_master_nccl_id
        self.gpus_per_node = gpus_per_node
        self.tokenize_on_node = tokenize_on_node

        # Will be initialized when training starts
        self._trainer = None
        self._train_loader = None
        self._eval_loader = None

    def setup(self) -> None:
        """Initialize data and model config on this node (non-blocking phase)."""
        # Import here to avoid loading CUDA on the driver
        from surogate.core.config.sft_config import SFTConfig
        from surogate.utils.dict import DictDefault
        from surogate.utils.logger import get_logger
        logger = get_logger()

        # Reconstruct config from dict (workers rebuild config objects locally)
        config = SFTConfig(DictDefault(self.config_dict))
        config.__post_init__()
        
        logger.info(f"Node {self.node_rank}: Model download complete, weights at {config.model_dir}")

        # Handle per-node tokenization if enabled
        if self.tokenize_on_node:
            train_files, eval_files = self._tokenize_node_data(config)
        else:
            train_files = self.train_files
            eval_files = self.eval_files

        # Store for init_trainer phase
        self._train_files = train_files
        self._eval_files = eval_files
        self._config = config

    def download_model(self) -> str:
        """
        Download the model (non-blocking phase that can take variable time per node).

        Returns:
            Path to the model weights file.
        """
        from surogate.utils.hf import get_model_weights_path
        from surogate.utils.logger import get_logger
        logger = get_logger()
        
        logger.info(f"Node {self.node_rank}: Starting model download...")
        # self.model_info, self.model_template, _, self.tokenizer = get_model_info_and_tokenizer(**self._config.get_model_kwargs(), load_model=False)
        # self.config.model_dir = self.model_info.model_dir
        model_weights_path = get_model_weights_path(self._config.model_dir)
        # logger.info(f"Node {self.node_rank}: Model download complete, weights at {model_weights_path}")

        # Store for later use
        self._model_weights_path = model_weights_path
        return model_weights_path

    def init_trainer(self) -> None:
        """
        Initialize the trainer with NCCL (collective operation - must be called synchronously across all nodes).

        IMPORTANT: download_model() must be called first and all nodes must complete their downloads
        before calling this method. The driver should use ray.get() to synchronize nodes between
        these two phases.
        """
        import os
        import time
        from surogate import _surogate
        from surogate.utils.tensor import to_surogate_dtype
        from surogate.utils.logger import get_logger
        logger = get_logger()
                
        # Ensure NCCL_DEBUG is set - use TRACE for maximum verbosity
        os.environ['NCCL_DEBUG'] = 'WARN'
        # os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        # Write NCCL debug output to a file for inspection
        # os.environ['NCCL_DEBUG_FILE'] = f'/tmp/nccl_debug_node_{self.node_rank}.log'

        # Disable InfiniBand/RoCE and force TCP sockets for cross-node communication
        # This is more reliable for debugging multi-node issues
        os.environ['NCCL_IB_DISABLE'] = '0'

        # Disable NCCL RAS (Reliability, Availability, Serviceability) subsystem
        # RAS can cause "connection closed by peer" issues during initialization
        # See: https://github.com/NVIDIA/nccl/issues/1718
        os.environ['NCCL_RAS_ENABLE'] = '0'

        # Configure NCCL network settings for multi-node communication
        # Auto-detect the correct network interface (not loopback)
        import socket
        import subprocess
        hostname = socket.gethostname()

        # Find network interface with a routable IP (not loopback)
        def get_network_interface():
            """Find the network interface and IP that can reach other nodes."""
            try:
                # Get all interfaces with their IPs
                result = subprocess.run(
                    ['ip', '-4', '-o', 'addr', 'show'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 4:
                        iface = parts[1]
                        ip_cidr = parts[3]
                        ip = ip_cidr.split('/')[0]
                        # Skip loopback and docker/veth interfaces
                        if iface == 'lo' or iface.startswith('docker') or iface.startswith('veth') or iface.startswith('br-'):
                            continue
                        # Prefer interfaces in common private ranges
                        if ip.startswith('192.168.') or ip.startswith('10.') or ip.startswith('172.'):
                            return iface, ip
                return None, None
            except Exception as e:
                logger.warning(f"Failed to detect network interface: {e}")
                return None, None

        detected_iface, detected_ip = get_network_interface()
        if detected_iface and 'NCCL_SOCKET_IFNAME' not in os.environ:
            os.environ['NCCL_SOCKET_IFNAME'] = detected_iface
            logger.info(f"Node {self.node_rank}: Auto-detected network interface {detected_iface} with IP {detected_ip}")

        local_ip = detected_ip or socket.gethostbyname(hostname)
        logger.info(f"Node {self.node_rank}: Local IP={local_ip}, hostname={hostname}, NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'NOT SET')}")

        logger.info(f"Node {self.node_rank}: Entering init_trainer at {time.time()}, NCCL_DEBUG={os.environ.get('NCCL_DEBUG', 'NOT SET')}, NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', 'NOT SET')}")

        # Use cached model weights path from download_model()
        model_weights_path = self._model_weights_path

        # Each node handles its share of the global batch
        # Local batch = per_device_batch * local_gpus * grad_accum
        local_gpus = self.gpus_per_node if self.gpus_per_node > 0 else self._config.gpus
        self.chunk_size = self._config.per_device_train_batch_size * self._config.sequence_len * local_gpus

        # Create data loaders
        # When tokenize_on_node=True, each node has its own complete shard,
        # so we use rank=0, world_size=1 (no further sharding needed)
        # When tokenize_on_node=False, we use strided access across pre-tokenized files
        if self.tokenize_on_node:
            self._train_loader = _surogate.DataLoader(
                self._train_files,
                self.chunk_size,
                rank=0,
                world_size=1,
                seed=self._config.train_seed
            )
            if self._eval_files:
                self._eval_loader = _surogate.DataLoader(
                    self._eval_files,
                    self.chunk_size,
                    rank=0,
                    world_size=1,
                    seed=self._config.eval_seed
                )
        else:
            # strided access across shared pre-tokenized files
            self._train_loader = _surogate.DataLoader(
                self._train_files,
                self.chunk_size,
                rank=self.node_rank,
                world_size=self.num_nodes,
                seed=self._config.train_seed
            )
            if self._eval_files:
                self._eval_loader = _surogate.DataLoader(
                    self._eval_files,
                    self.chunk_size,
                    rank=self.node_rank,
                    world_size=self.num_nodes,
                    seed=self._config.eval_seed
                )

        # Create model config
        from surogate.dsl.ir_builder import build_dsl_ir_for_model
        ir_json = build_dsl_ir_for_model(self._config.model_dir)
        self._config.runtime_config.dsl_ir_json = ir_json
        pretrained_config = _surogate.PretrainedConfig.from_pretrained(
            self._config.model_dir, to_surogate_dtype(self._config.torch_dtype)
        )

        # Determine if using LoRA
        use_lora = self._config.lora and self._config.lora_rank and self._config.lora_alpha and self._config.lora_target_modules

        # Check for checkpoint resumption
        self.start_step = 0
        if self._config.resume_from_checkpoint:
            self.start_step = _surogate.find_latest_checkpoint(self._config.checkpoint_dir)
            if self.start_step >= 0:
                logger.info(f"Node {self.node_rank}: Found checkpoint at step {self.start_step}")
            else:
                logger.warning(f"Node {self.node_rank}: No checkpoint found to resume from. Starting training from beginning.")
                self.start_step = 0

        # Create the trainer with NCCL multi-node support
        if self.num_nodes > 1:
            # Synchronization barrier: ensure all nodes are ready before NCCL initialization
            # Sleep for a small amount to ensure all nodes reach this point
            logger.info(f"Node {self.node_rank}: Ready for NCCL initialization, waiting for other nodes...")
            logger.info(f"Node {self.node_rank}: NCCL ID (first 16 bytes): {self.nccl_id[:16].hex()}")
            logger.info(f"Node {self.node_rank}: Node Master NCCL ID (first 16 bytes): {self.node_master_nccl_id[:16].hex()}")
            time.sleep(2.0)  # Give all nodes time to reach this point

            # Multi-node: use NCCL IDs for cross-node coordination
            logger.info(f"Node {self.node_rank}: Starting NCCL initialization with {self.num_nodes} nodes, node_rank={self.node_rank}, local_gpus={local_gpus}")
            self._trainer = _surogate.SurogateTrainer.create_multinode(
                ngpu=local_gpus,
                node_rank=self.node_rank,
                num_nodes=self.num_nodes,
                nccl_id=self.nccl_id,
                node_master_nccl_id=self.node_master_nccl_id,
                config=pretrained_config,
                options=self._config.runtime_config,
                batch_size=self._config.per_device_train_batch_size,
                seq_len=self._config.sequence_len,
                grad_accum=self._config.gradient_accumulation_steps,
                memcpy_all_gather=self._config.memcpy_all_gather,
                memcpy_send_recv=self._config.memcpy_send_recv,
                lora_config=self._config.lora_config if use_lora else None,
                qlora_config=self._config.qlora_config if use_lora else None
            )
            logger.info(f"Node {self.node_rank}: NCCL initialization completed successfully")
        else:
            # Single-node: use standard constructor
            self._trainer = _surogate.SurogateTrainer(
                ngpu=local_gpus,
                config=pretrained_config,
                options=self._config.runtime_config,
                batch_size=self._config.per_device_train_batch_size,
                seq_len=self._config.sequence_len,
                grad_accum=self._config.gradient_accumulation_steps,
                memcpy_all_gather=self._config.memcpy_all_gather,
                memcpy_send_recv=self._config.memcpy_send_recv,
                lora_config=self._config.lora_config if use_lora else None,
                qlora_config=self._config.qlora_config if use_lora else None
            )

        # Load checkpoint or import weights
        if self._config.resume_from_checkpoint and self.start_step >= 0:
            # Base model weights must be imported first to initialize the weight structure.
            # For LoRA: checkpoint only contains adapter weights + optimizer state, so we
            #           need the original base model weights.
            # For FFT/upcycle: checkpoint contains trained weights. We import from the
            #                  checkpoint's model.safetensors to handle upcycled models
            #                  where config.model_dir points to a different architecture.
            if use_lora:
                # LoRA: use base model weights
                weights_path = model_weights_path
            else:
                # FFT/upcycle: use checkpoint's saved weights (handles architecture changes)
                checkpoint_dir = Path(self._config.checkpoint_dir) / f"step_{self.start_step:08d}"
                checkpoint_weights = checkpoint_dir / "model.safetensors"
                if checkpoint_weights.exists():
                    weights_path = str(checkpoint_weights)
                else:
                    # Fallback to base model if checkpoint doesn't have model.safetensors
                    weights_path = model_weights_path
            logger.info(f"Node {self.node_rank}: Importing base model weights from {weights_path}...")
            self._trainer.import_weights(weights_path)
            logger.info(f"Node {self.node_rank}: Loading checkpoint from step {self.start_step}...")
            self._trainer.load_checkpoint(str(self._config.checkpoint_dir), self.start_step)
            logger.info(f"Node {self.node_rank}: Checkpoint loaded successfully")
        else:
            # Import weights from pretrained model
            self._trainer.import_weights(model_weights_path)

        self.local_gpus = local_gpus

        logger.info(f"Node {self.node_rank}: Completed init_trainer at {time.time()}")

    def _tokenize_node_data(self, config: "SFTConfig") -> Tuple[List[str], Optional[List[str]]]:
        """
        Tokenize this node's shard of the dataset.

        Each node tokenizes only 1/num_nodes of the training data, while all nodes
        get the full validation dataset for consistent evaluation metrics.

        Args:
            config: The SFTConfig object (already reconstructed from dict).

        Returns:
            Tuple of (train_files, eval_files) paths for this node.
        """
        import tempfile
        from surogate.train.tokenize import TokenizeDatasets
        from surogate.utils.dict import DictDefault
        from surogate.utils.logger import get_logger
        logger = get_logger()
        
        # Determine base output directory for tokenized data on this worker
        # Priority: distributed.worker_output_dir > /tmp/surogate-{run_name}
        if config.distributed and config.distributed.worker_output_dir:
            base_output_dir = config.distributed.worker_output_dir
        else:
            # Use temp directory with run_name for reproducibility across restarts
            base_output_dir = os.path.join(tempfile.gettempdir(), f"surogate-{config.run_name}")

        # Create node-specific subdirectory
        node_output_dir = os.path.join(base_output_dir, f"node-{self.node_rank}")
        os.makedirs(node_output_dir, exist_ok=True)
        logger.info(f"Node {self.node_rank}: Using output directory {node_output_dir}")

        # Set node sharding info on config for the tokenizer to use
        config._node_rank = self.node_rank
        config._num_nodes = self.num_nodes
        config.output_dir = node_output_dir

        # Check if tokenization is needed (hash-based caching works per-node)
        train_files = sorted(glob.glob(os.path.join(node_output_dir, "train*.bin")))

        if not train_files:
            logger.info(f"Node {self.node_rank}: Tokenizing dataset shard ({1}/{self.num_nodes})...")
            tokenizer = TokenizeDatasets(config, args=DictDefault({}))
            tokenizer.run()

            # Get the files that were written
            train_files = sorted(glob.glob(os.path.join(node_output_dir, "train*.bin")))
            logger.info(f"Node {self.node_rank}: Tokenization complete. {len(train_files)} train file(s) created.")
        else:
            logger.info(f"Node {self.node_rank}: Using cached tokenized data ({len(train_files)} train file(s)).")

        # Get eval files
        eval_files = sorted(glob.glob(os.path.join(node_output_dir, "eval*.bin")))
        if not eval_files:
            eval_files = None

        return train_files, eval_files

    def train_step(
        self,
        step: int,
        lr: float,
    ) -> Tuple[float, float]:
        """
        Execute one training step on this node.

        Returns:
            Tuple of (loss, grad_norm) from this node.
        """
        from surogate import _surogate

        config = self._config
        B = config.per_device_train_batch_size
        T = config.sequence_len
        local_gpus = self.local_gpus
        use_full_step_graphs = True
        if use_full_step_graphs and config.optimizer not in ("adamw_8bit", "normuon"):
            raise RuntimeError("DSL training requires optimizer 'adamw_8bit' or 'normuon' for full-step execution.")
        if use_full_step_graphs and not config.use_cuda_graphs:
            logger.info("CUDA graphs disabled; DSL full-step execution will use eager fallback.")

        # Allocate token buffers
        micro_steps = config.gradient_accumulation_steps if use_full_step_graphs else 1
        total_rows = local_gpus * B * micro_steps
        in_tokens = np.empty((total_rows, T), dtype=np.int32)
        out_tokens = np.empty((total_rows, T), dtype=np.int32)
        pos_ids = np.empty((total_rows, T), dtype=np.int32)

        if use_full_step_graphs:
            chunk = local_gpus * B
            for micro_step in range(config.gradient_accumulation_steps):
                if not self._train_loader.has_next():
                    self._train_loader.advance_epoch()
                start = micro_step * chunk
                end = start + chunk
                self._train_loader.load_batch(in_tokens[start:end], out_tokens[start:end], pos_ids[start:end])
        else:
            # Run gradient accumulation steps
            for micro_step in range(config.gradient_accumulation_steps):
                if not self._train_loader.has_next():
                    self._train_loader.advance_epoch()

                self._train_loader.load_batch(in_tokens, out_tokens, pos_ids)
                self._trainer.step(in_tokens, out_tokens, pos_ids)

        # Optimizer update
        opt_config = _surogate.OptimizerConfig(
            optimizer=config.optimizer,
            learning_rate=lr,
            weight_decay=config.weight_decay,
            grad_clip=config.max_grad_norm,
            adamw_beta1=config.adamw_beta1,
            adamw_beta2=config.adamw_beta2,
            adamw_epsilon=config.adamw_epsilon,
            normuon_momentum=config.normuon_momentum,
            normuon_beta2=config.normuon_beta2,
            normuon_lr=lr,
            normuon_cautious_wd=config.normuon_cautious_wd
        )
        if use_full_step_graphs:
            result = self._trainer.train_step_graphed(in_tokens, out_tokens, pos_ids, opt_config, step + 1)
        else:
            result = self._trainer.update_with_config(opt_config, step + 1)

        return result['loss'], result['norm']

    def get_moe_stats(self) -> Dict[str, Any]:
        """Get MoE training statistics from the last forward pass."""
        if self._trainer is not None:
            return self._trainer.get_moe_stats()
        return {'valid': False}

    def validate(self, max_steps: int = 100) -> float:
        """Run validation and return mean loss."""
        if not self._eval_loader:
            return 0.0

        config = self._config
        B = config.per_device_train_batch_size
        T = config.sequence_len
        local_gpus = self.local_gpus

        in_tokens = np.empty((local_gpus * B, T), dtype=np.int32)
        out_tokens = np.empty((local_gpus * B, T), dtype=np.int32)
        pos_ids = np.empty((local_gpus * B, T), dtype=np.int32)

        self._eval_loader.set_state(self._eval_loader.seed, 0, 0, 0)
        total_loss = 0.0
        batches = 0

        while self._eval_loader.has_next() and (max_steps < 0 or batches < max_steps):
            self._eval_loader.load_batch(in_tokens, out_tokens, pos_ids)
            loss = self._trainer.validate(in_tokens, out_tokens, pos_ids)
            total_loss += loss
            batches += 1

        return total_loss / batches if batches > 0 else 0.0

    def save_checkpoint(self, path: str, step: int) -> None:
        """Save checkpoint (only node 0 saves the full model)."""
        self._trainer.save_checkpoint(path, step)

    def cleanup_trainer(self) -> None:
        """Cleanup trainer resources before export."""
        # Release NCCL/CUDA resources by deleting the trainer
        # This is needed to prevent hangs during export
        if self._trainer is not None:
            del self._trainer
            self._trainer = None

    def export_model(self, path: str) -> bool:
        """Export the model.

        NOTE: ALL nodes must call this method because the C++ export_model
        may contain NCCL barriers that require all ranks to participate.
        Only node 0 actually writes the file, but all nodes must participate
        in any synchronization.
        """
        from surogate.utils.logger import get_logger
        logger = get_logger()
        logger.info(f"Node {self.node_rank}: export_model called")
        if self._trainer is not None:
            logger.info(f"Node {self.node_rank}: Starting model export (writing={self.node_rank == 0})")
            self._trainer.export_model(path)
            logger.info(f"Node {self.node_rank}: Model export complete")
            return self.node_rank == 0  # Only node 0 actually writes the file
        return False

    def export_adapter(self, path: str) -> bool:
        """Export LoRA adapter.

        NOTE: ALL nodes must call this method because the C++ export_adapter
        contains NCCL barriers that require all ranks to participate.
        Only node 0 actually writes the file, but all nodes must participate
        in the barrier synchronization.
        """
        from surogate.utils.logger import get_logger
        logger = get_logger()
        logger.info(f"Node {self.node_rank}: export_adapter called")
        if self._trainer is not None:
            logger.info(f"Node {self.node_rank}: Starting adapter export (writing={self.node_rank == 0})")
            self._trainer.export_adapter(path)
            logger.info(f"Node {self.node_rank}: Adapter export complete")
            return self.node_rank == 0  # Only node 0 actually writes the file
        return False

class RayDistributedTrainer:
    """
    Ray-based distributed trainer for multi-node training.

    Spawns one Ray actor per node, each using the threaded backend for local GPUs.
    Cross-node gradient synchronization is handled via NCCL (uses InfiniBand when available).
    """

    def __init__(
        self,
        config: "SFTConfig",
        train_files: List[str],
        eval_files: Optional[List[str]] = None,
        ray_address: str = "auto",
        num_nodes: Optional[int] = None,
        gpus_per_node: int = 0,  # 0 = use config.gpus
        tokenize_on_node: bool = False,  # If True, each node tokenizes its own data shard
    ):
        """
        Initialize the distributed trainer.

        Args:
            config: Training configuration.
            train_files: List of training data files. May be empty if tokenize_on_node=True.
            eval_files: Optional list of evaluation data files. May be empty if tokenize_on_node=True.
            ray_address: Ray cluster address ("auto", "local", or "ray://host:port").
            num_nodes: Number of nodes to use (default: auto-detect from Ray cluster).
            gpus_per_node: GPUs per node (0 = use config.gpus).
            tokenize_on_node: If True, each node loads and tokenizes its own 1/num_nodes
                shard of the dataset instead of using pre-tokenized files. This
                reduces driver memory pressure and enables parallel tokenization.
        """
        ray = _get_ray()

        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(address=ray_address if ray_address != "local" else None)

        self._config = config
        self.train_files = train_files
        self.eval_files = eval_files
        self.gpus_per_node = gpus_per_node if gpus_per_node > 0 else config.gpus
        self.tokenize_on_node = tokenize_on_node

        # Determine number of nodes
        if num_nodes is None:
            # Auto-detect from Ray cluster
            nodes = ray.nodes()
            num_nodes = len([n for n in nodes if n.get('Alive', False)])

        self.num_nodes = num_nodes
        self.node_trainers: List[ray.actor.ActorHandle] = []

        # Create serializable config dict (exclude non-serializable C++ objects)
        # Workers will reconstruct runtime_config, lora_config, qlora_config from the dict
        config_dict = {}
        for key, value in config.__dict__.items():
            # Skip non-serializable C++ objects (will be reconstructed on workers)
            if key in ('runtime_config', 'lora_config', 'qlora_config'):
                continue
            # Recursively serialize the value
            serialized = _serialize_config_value(value)
            if serialized is not None:
                config_dict[key] = serialized
        self.config_dict = config_dict

        # Generate NCCL IDs on the driver (will be shared with all workers)
        # Import here to avoid loading CUDA on the driver before Ray actors are created
        from surogate import _surogate
        self.nccl_id = _surogate.generate_nccl_id()
        self.node_master_nccl_id = _surogate.generate_nccl_id()

    def _setup_workers(self) -> None:
        from surogate.utils.logger import get_logger
        logger = get_logger()
        
        """Create Ray actors for each node."""
        ray = _get_ray()

        # Create one actor per node with GPU resources
        gpus_per_node = self.gpus_per_node

        @ray.remote(
            num_gpus=gpus_per_node,
            runtime_env={"env_vars": {"NCCL_DEBUG": "INFO"}}
        )
        class NodeTrainerActor:
            def __init__(
                self,
                config_dict: Dict[str, Any],
                train_files: List[str],
                eval_files: Optional[List[str]],
                node_rank: int,
                num_nodes: int,
                nccl_id: bytes,
                node_master_nccl_id: bytes,
                gpus_per_node: int,
                tokenize_on_node: bool = False,
            ):  
                self.trainer = NodeTrainer(
                    config_dict=config_dict,
                    train_files=train_files,
                    eval_files=eval_files,
                    node_rank=node_rank,
                    num_nodes=num_nodes,
                    nccl_id=nccl_id,
                    node_master_nccl_id=node_master_nccl_id,
                    gpus_per_node=gpus_per_node,
                    tokenize_on_node=tokenize_on_node,
                )

            def setup(self) -> None:
                self.trainer.setup()

            def download_model(self) -> str:
                return self.trainer.download_model()

            def wait_at_barrier(self, barrier_actor) -> None:
                """Wait at a barrier until all nodes arrive (used before NCCL init)."""
                import ray
                import time

                # Register arrival and get the generation we're waiting for
                my_gen, count = ray.get(barrier_actor.arrive.remote())

                # Poll until generation changes (meaning all participants arrived)
                while ray.get(barrier_actor.get_generation.remote()) == my_gen:
                    time.sleep(0.001)  # 1ms poll

            def init_trainer(self) -> None:
                self.trainer.init_trainer()

            def get_start_step(self) -> int:
                """Get the starting step (0 for fresh training, >0 for resumed from checkpoint)."""
                return self.trainer.start_step

            def train_step(self, step: int, lr: float) -> Tuple[float, float]:
                return self.trainer.train_step(step, lr)

            def validate(self, max_steps: int = 100) -> float:
                return self.trainer.validate(max_steps)

            def save_checkpoint(self, path: str, step: int) -> None:
                self.trainer.save_checkpoint(path, step)

            def export_model(self, path: str) -> bool:
                return self.trainer.export_model(path)

            def export_adapter(self, path: str) -> bool:
                return self.trainer.export_adapter(path)

            def get_num_tokens(self) -> int:
                """Get the number of tokens in the training dataset."""
                return self.trainer._train_loader.num_tokens

            def get_moe_stats(self) -> Dict[str, Any]:
                """Get MoE training statistics from the last forward pass."""
                return self.trainer.get_moe_stats()

        # Spawn actors with shared NCCL IDs
        self.node_trainers = [
            NodeTrainerActor.remote(
                config_dict=self.config_dict,
                train_files=self.train_files,
                eval_files=self.eval_files,
                node_rank=i,
                num_nodes=self.num_nodes,
                nccl_id=self.nccl_id,
                node_master_nccl_id=self.node_master_nccl_id,
                gpus_per_node=gpus_per_node,
                tokenize_on_node=self.tokenize_on_node,
            )
            for i in range(self.num_nodes)
        ]

        # Phase 1: Setup (tokenization, data loading) - can take different times per node
        ray.get([t.setup.remote() for t in self.node_trainers])

        # Phase 2: Download models - can take different times per node (network-bound)
        # This MUST complete on all nodes before NCCL initialization
        logger.info("Downloading models on all nodes...")
        ray.get([t.download_model.remote() for t in self.node_trainers])
        logger.info("All nodes finished downloading models")

        # Phase 3: Barrier - ensure all nodes are ready to enter NCCL init together
        # NCCL's ncclCommInitRank is a collective operation that requires all participants
        # to call it at nearly the same time. Without this barrier, Ray scheduling delays
        # could cause one node to start NCCL init while another is still finishing download.
        if self.num_nodes > 1:
            logger.info("Synchronizing nodes before NCCL initialization...")
            barrier = _create_barrier_actor(self.num_nodes)
            ray.get([t.wait_at_barrier.remote(barrier) for t in self.node_trainers])
            logger.info("All nodes synchronized, proceeding to NCCL init")

        # Phase 4: Initialize trainers with NCCL (collective operation - must be synchronous)
        # All nodes must enter this phase together since it contains NCCL collective operations
        logger.info("Initializing NCCL trainers on all nodes...")
        ray.get([t.init_trainer.remote() for t in self.node_trainers])
        logger.info("All nodes finished NCCL initialization")

        # Get start step from node 0 (all nodes should have the same value)
        start_step = ray.get(self.node_trainers[0].get_start_step.remote())
        return start_step

    def train(self) -> None:
        """Run the distributed training loop."""
        ray = _get_ray()
        from surogate.train.loss_guard import LossGuard
        from surogate.train.lr_schedule import LRSchedule
        from surogate.utils.logger import get_logger

        logger = get_logger()

        # Setup workers
        logger.info(f"Setting up distributed training with {self.num_nodes} nodes...")
        start_step = self._setup_workers()

        # Calculate training parameters
        config = self._config
        local_gpus = self.gpus_per_node
        tokens_per_step_per_node = (
            config.per_device_train_batch_size *
            config.sequence_len *
            local_gpus *
            config.gradient_accumulation_steps
        )
        total_tokens_per_step = tokens_per_step_per_node * self.num_nodes

        # Determine max steps
        # Note: In distributed mode, each node sees 1/num_nodes of the data (sharded via strided access)
        if config.max_steps > 0:
            max_steps = config.max_steps
        else:
            # Calculate steps per epoch from the data loader
            # Get the first node's loader info to determine total dataset size
            num_tokens = ray.get(self.node_trainers[0].get_num_tokens.remote())
            steps_per_epoch = num_tokens // total_tokens_per_step
            max_steps = steps_per_epoch * config.num_epochs
            logger.info(f"Calculated {steps_per_epoch} steps per epoch from {num_tokens} tokens")

        # Apply warmup_ratio if warmup_steps is 0
        warmup_steps = config.warmup_steps
        if warmup_steps == 0 and config.warmup_ratio > 0:
            warmup_steps = int(max_steps * config.warmup_ratio)
            logger.info(f"Derived {warmup_steps} warmup steps from warmup_ratio={config.warmup_ratio}")

        # Learning rate schedule
        lr_schedule = LRSchedule(
            base_lr=config.learning_rate,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            cooldown_steps=config.cooldown_steps,
            final_lr=config.learning_rate * config.final_lr_fraction,
            schedule_type=config.lr_scheduler_type
        )

        # Auto LR reduction guard
        loss_guard = LossGuard(lr_schedule, logger) if config.auto_lr_reduction else None

        logger.info(f"Starting distributed training...")
        logger.info(f"  Nodes: {self.num_nodes}")
        logger.info(f"  GPUs per node: {local_gpus}")
        logger.info(f"  Total GPUs: {self.num_nodes * local_gpus}")
        logger.info(f"  Tokens per step: {total_tokens_per_step}")
        logger.info(f"  Starting from step: {start_step}")
        logger.info(f"  Max steps: {max_steps}")

        # Training loop
        import time
        step_start_time = time.time()

        for step in range(start_step, max_steps):
            lr = lr_schedule.get_lr(step)

            # Run training step on all nodes in parallel
            futures = [t.train_step.remote(step, lr) for t in self.node_trainers]
            results = ray.get(futures)

            # Aggregate results (average loss and norm across nodes)
            losses = [r[0] for r in results]
            norms = [r[1] for r in results]
            avg_loss = sum(losses) / len(losses)
            avg_norm = sum(norms) / len(norms)

            # Check for loss spikes / gradient explosions
            if loss_guard is not None:
                loss_guard.step(avg_loss, avg_norm, step)

            # Calculate timing and throughput
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            tokens_per_sec = total_tokens_per_step / step_time if step_time > 0 else 0

            # Log progress
            logger.info(
                f"Step {step}/{max_steps} | Loss: {avg_loss:.4f} | Norm: {avg_norm:.4f} | "
                f"LR: {lr:.2e} | {step_time:.2f}s | {tokens_per_sec:.0f} tok/s"
            )

            # Log MoE stats for MoE models (get from node 0)
            moe_stats = ray.get(self.node_trainers[0].get_moe_stats.remote())
            if moe_stats.get('valid', False):
                logger.info(
                    f"  MoE: aux_loss={moe_stats['aux_loss']:.4f} z_loss={moe_stats['z_loss']:.4f} "
                    f"util={moe_stats['expert_utilization']:.2%} imbalance={moe_stats['load_imbalance']:.2f}"
                )

            # Reset timer for next step
            step_start_time = time.time()

            # Periodic evaluation
            if self.eval_files and config.eval_steps > 0 and step % config.eval_steps == 0 and step > start_step:
                eval_futures = [t.validate.remote(100) for t in self.node_trainers]
                eval_losses = ray.get(eval_futures)
                avg_eval_loss = sum(eval_losses) / len(eval_losses)
                logger.info(f"  Eval loss: {avg_eval_loss:.4f}")

            # Periodic checkpointing
            if config.save_steps > 0 and step % config.save_steps == 0 and step > start_step:
                logger.info(f"Saving checkpoint at step {step}...")
                try:
                    # Only node 0 saves (others have identical weights in data parallel)
                    ray.get(self.node_trainers[0].save_checkpoint.remote(config.checkpoint_dir, step))
                    logger.info(f"Checkpoint saved successfully at step {step}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {step}: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.warning("Training will continue without saving this checkpoint")
                    import traceback
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Save final model
        # IMPORTANT: export_adapter/export_model contain NCCL barriers, so ALL nodes must participate
        logger.info("Training complete. Saving final model...")
        try:
            if config.lora:
                adapter_dir = str(Path(config.output_dir))
                logger.info(f"Exporting LoRA adapter to {adapter_dir}...")
                # Call export on ALL nodes - they all participate in NCCL barriers
                # Only node 0 actually writes the file
                export_refs = [t.export_adapter.remote(adapter_dir) for t in self.node_trainers]
                ready, not_ready = ray.wait(export_refs, num_returns=len(export_refs), timeout=120)
                if len(ready) == len(export_refs):
                    results = ray.get(ready)
                    if any(results):
                        logger.info(f"LoRA adapter saved to {adapter_dir}")
                    else:
                        logger.warning("Adapter export: all nodes returned False")
                else:
                    # Check if file was saved despite timeout
                    adapter_file = Path(adapter_dir) / "adapter_model.safetensors"
                    if adapter_file.exists():
                        logger.info(f"LoRA adapter saved to {adapter_dir} (export timed out but file exists)")
                    else:
                        logger.warning(f"Export timed out after 120s. {len(ready)}/{len(export_refs)} nodes completed.")

                # Merge adapter into base model if requested (only on head node)
                if config.merge_adapter:
                    from surogate.utils.adapter_merge import merge_adapter
                    merged_dir = Path(config.output_dir)
                    try:
                        merge_adapter(
                            base_model_path=config.model_dir,
                            adapter_path=adapter_dir,
                            output_path=str(merged_dir),
                            max_shard_size="5GB",
                            cpu_offload=True
                        )
                    except Exception as e:
                        logger.error(f"Failed to merge adapter: {e}")
                        import traceback
                        logger.error(f"Traceback:\n{traceback.format_exc()}")
                        logger.warning("Adapter merge failed, but adapter was saved successfully")
            else:
                logger.info(f"Exporting model to {config.output_dir}...")
                # Call export on ALL nodes - they all participate in NCCL barriers
                # Only node 0 actually writes the file
                export_refs = [t.export_model.remote(config.output_dir) for t in self.node_trainers]
                ready, not_ready = ray.wait(export_refs, num_returns=len(export_refs), timeout=120)
                if len(ready) == len(export_refs):
                    results = ray.get(ready)
                    if any(results):
                        logger.info(f"Model saved to {config.output_dir}")
                        # Copy tokenizer files from source model
                        self._copy_tokenizer_files(config.model_dir, config.output_dir)
                    else:
                        logger.warning("Model export: all nodes returned False")
                else:
                    # Check if file was saved despite timeout
                    model_file = Path(config.output_dir) / "model.safetensors"
                    if model_file.exists():
                        logger.info(f"Model saved to {config.output_dir} (export timed out but file exists)")
                        # Copy tokenizer files from source model
                        self._copy_tokenizer_files(config.model_dir, config.output_dir)
                    else:
                        logger.warning(f"Export timed out after 120s. {len(ready)}/{len(export_refs)} nodes completed.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        finally:
            # Cleanup Ray actors
            logger.info("Shutting down Ray actors...")
            self.shutdown()
            logger.info("Ray actors shut down. Training complete.")

    def _copy_tokenizer_files(self, src_dir: str, dst_dir: str):
        """Copy tokenizer and vocab files from source model to output directory."""
        from surogate.utils.logger import get_logger
        logger = get_logger()
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt"
        ]
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        for filename in tokenizer_files:
            src = src_path / filename
            if src.exists():
                shutil.copy(src, dst_path / filename)
                logger.info(f"Copied {filename}")

    def shutdown(self) -> None:
        """Shutdown Ray actors and cleanup resources."""
        ray = _get_ray()

        # Kill actors forcefully to release CUDA/NCCL resources
        for t in self.node_trainers:
            try:
                ray.kill(t, no_restart=True)
            except Exception:
                pass  # Ignore errors during shutdown

        self.node_trainers = []
