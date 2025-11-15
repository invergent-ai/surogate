# surogate/eval/stress/resource_monitor.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
import threading
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float

    # GPU metrics (optional)
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_memory_total_mb': self.gpu_memory_total_mb,
            'gpu_utilization_percent': self.gpu_utilization_percent,
            'gpu_temperature': self.gpu_temperature,
        }


class ResourceMonitor:
    """Monitor system resources during stress testing."""

    def __init__(self, interval: float = 0.5):
        """
        Initialize resource monitor.

        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Check for optional dependencies
        self.has_psutil = self._check_psutil()
        self.has_nvidia = self._check_nvidia_smi()
        self.has_pynvml = self._check_pynvml()

    def _check_psutil(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil
            return True
        except ImportError:
            logger.warning("psutil not installed - CPU/memory monitoring disabled")
            logger.warning("Install with: pip install psutil")
            return False

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False

    def _check_pynvml(self) -> bool:
        """Check if pynvml is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except:
            if self.has_nvidia:
                logger.warning("pynvml not installed - GPU monitoring limited")
                logger.warning("Install with: pip install nvidia-ml-py3")
            return False

    def start(self):
        """Start monitoring resources."""
        if not self.has_psutil:
            logger.warning("Resource monitoring disabled (psutil not available)")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop(self):
        """Stop monitoring resources."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info(f"Resource monitoring stopped ({len(self.snapshots)} snapshots collected)")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error capturing resource snapshot: {e}")

    def _capture_snapshot(self) -> ResourceSnapshot:
        """Capture current resource state."""
        import psutil

        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # GPU metrics
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        gpu_temperature = None

        if self.has_pynvml:
            try:
                gpu_memory_used, gpu_memory_total, gpu_utilization, gpu_temperature = self._get_gpu_metrics_pynvml()
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {e}")
        elif self.has_nvidia:
            try:
                gpu_memory_used, gpu_memory_total, gpu_utilization = self._get_gpu_metrics_nvidia_smi()
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {e}")

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization_percent=gpu_utilization,
            gpu_temperature=gpu_temperature,
        )

    def _get_gpu_metrics_pynvml(self):
        """Get GPU metrics using pynvml."""
        import pynvml

        # Get first GPU (can be extended for multi-GPU)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used = mem_info.used / (1024 * 1024)  # MB
        gpu_memory_total = mem_info.total / (1024 * 1024)  # MB

        # Utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu

        # Temperature
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        return gpu_memory_used, gpu_memory_total, gpu_utilization, temperature

    def _get_gpu_metrics_nvidia_smi(self):
        """Get GPU metrics using nvidia-smi."""
        import subprocess

        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            values = result.stdout.strip().split(',')
            gpu_memory_used = float(values[0])
            gpu_memory_total = float(values[1])
            gpu_utilization = float(values[2])
            return gpu_memory_used, gpu_memory_total, gpu_utilization

        return None, None, None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of monitoring session."""
        if not self.snapshots:
            return {}

        # CPU stats
        cpu_values = [s.cpu_percent for s in self.snapshots]

        # Memory stats
        memory_values = [s.memory_percent for s in self.snapshots]
        memory_used = [s.memory_used_mb for s in self.snapshots]

        summary = {
            'duration_seconds': self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            'num_snapshots': len(self.snapshots),
            'cpu': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values),
            },
            'memory': {
                'min_percent': min(memory_values),
                'max_percent': max(memory_values),
                'avg_percent': sum(memory_values) / len(memory_values),
                'min_used_mb': min(memory_used),
                'max_used_mb': max(memory_used),
                'avg_used_mb': sum(memory_used) / len(memory_used),
            }
        }

        # GPU stats (if available)
        gpu_memory_values = [s.gpu_memory_used_mb for s in self.snapshots if s.gpu_memory_used_mb is not None]
        gpu_util_values = [s.gpu_utilization_percent for s in self.snapshots if s.gpu_utilization_percent is not None]

        if gpu_memory_values:
            summary['gpu'] = {
                'memory_min_mb': min(gpu_memory_values),
                'memory_max_mb': max(gpu_memory_values),
                'memory_avg_mb': sum(gpu_memory_values) / len(gpu_memory_values),
            }

        if gpu_util_values:
            summary['gpu'] = summary.get('gpu', {})
            summary['gpu'].update({
                'utilization_min': min(gpu_util_values),
                'utilization_max': max(gpu_util_values),
                'utilization_avg': sum(gpu_util_values) / len(gpu_util_values),
            })

        return summary