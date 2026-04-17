import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import psutil

from surogate import _surogate
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class GPUInfo:
    device_id: int
    name: str
    total_memory: int
    compute_capability_major: int
    compute_capability_minor: int


gpu_info: list[GPUInfo] = _surogate.SystemInfo.get_gpu_info()


def gpu_count() -> int:
    return len(gpu_info)


def cuda_is_available() -> bool:
    return (
        _surogate.SystemInfo.get_cuda_driver_version() is not None
        and _surogate.SystemInfo.get_cuda_runtime_version() is not None
        and gpu_count() > 0
    )


def get_system_info() -> dict[str, Any]:
    info = {
        "python_version": sys.version,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor(),
    }

    if gpu_count() > 0:
        try:
            info["cuda_runtime_version"] = _surogate.SystemInfo.get_cuda_runtime_version()
            info["cuda_driver_version"] = _surogate.SystemInfo.get_cuda_driver_version()
            info["nccl_version"] = _surogate.SystemInfo.get_nccl_version()
            info["cudnn_version"] = _surogate.SystemInfo.get_cudnn_version()

            info["gpu_count"] = len(gpu_info)
            info["gpu_name"] = gpu_info[0].name
            info["gpu_memory_gb"] = gpu_info[0].total_memory / 1e9

            # Detailed GPU info for first device
            compute_cap = (gpu_info[0].compute_capability_major, gpu_info[0].compute_capability_minor)
            info["gpu_compute_capability"] = f"{compute_cap[0]}.{compute_cap[1]}"
        except Exception as e:
            info["cuda_error"] = str(e)

    # System memory
    try:
        import psutil

        memory = psutil.virtual_memory()
        info["system_memory_gb"] = memory.total / 1e9
        info["available_memory_gb"] = memory.available / 1e9
        info["memory_percent_used"] = memory.percent
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)

        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info["cpu_freq_current_mhz"] = cpu_freq.current
                info["cpu_freq_min_mhz"] = cpu_freq.min
                info["cpu_freq_max_mhz"] = cpu_freq.max
        except:
            pass

        # CPU usage
        try:
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        except:
            pass
    except ImportError:
        info["psutil_available"] = False

    # Disk space
    try:
        disk_usage = shutil.disk_usage(".")
        info["disk_space_gb"] = disk_usage.free / 1e9
        info["disk_total_gb"] = disk_usage.total / 1e9
        info["disk_used_gb"] = disk_usage.used / 1e9
        info["disk_percent_used"] = (disk_usage.used / disk_usage.total) * 100
    except Exception:
        pass

    return info


def print_system_diagnostics(system_info):
    logger.header("System Information")
    if cuda_is_available():
        logger.metrics(
            names=["CUDA", "Driver", "NCCL", "cuDNN", "GPU Count"],
            values=[
                system_info.get("cuda_runtime_version", "Unknown"),
                system_info.get("cuda_driver_version", "Unknown"),
                system_info.get("nccl_version", "Unknown"),
                system_info.get("cudnn_version", "Unknown"),
                system_info.get("gpu_count", 0),
            ],
        )
        names = []
        values = []
        for i in range(gpu_count()):
            props = gpu_info[i]
            compute_cap = (props.compute_capability_major, props.compute_capability_minor)
            names.append(f"GPU{i}")
            values.append(f"{props.name}, sm_{compute_cap[0]}.{compute_cap[1]}, {props.total_memory / 1e9:.2f}GB")
        logger.metrics(names=names, values=values)
    else:
        logger.warning("WARNING: CUDA not available, using CPU only.")

    if system_info.get("system_memory_gb"):
        total_mem = system_info.get("system_memory_gb", 0)
        avail_mem = system_info.get("available_memory_gb", 0)
        logger.metrics(
            names=["Available Memory", "Total Memory"],
            values=[f"{avail_mem:.2f}", f"{total_mem:.2f}"],
            units=["GB", "GB"],
        )

    if system_info.get("cpu_count"):
        metric_names = ["CPU"]
        metric_values = [str(system_info.get("cpu_count", 0))]
        metric_units = ["cores"]
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metric_names.append("CPU Freq)")
                metric_values.append(f"{cpu_freq.current:.0f}")
                metric_units.append("MHz")
        except:
            pass

        logger.metrics(names=metric_names, values=metric_values, units=metric_units)

    if system_info.get("disk_space_gb"):
        disk_free = system_info.get("disk_space_gb", 0)
        try:
            disk_usage = shutil.disk_usage(".")
            disk_total = disk_usage.total / 1e9
            logger.metrics(
                names=["Disk Free", "Disk Total"], values=[f"{disk_free:.2f}", f"{disk_total:.2f}"], units=["GB", "GB"]
            )
        except:
            logger.metric(f"Disk: {disk_free:.2f} GB free")
            logger.metrics(names=["Disk Free"], values=[f"{disk_free:.2f}"], units=["GB"])
