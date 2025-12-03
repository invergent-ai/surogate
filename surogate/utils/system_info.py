import platform
import shutil
from typing import Any, Dict

import psutil
import torch
import sys

from surogate.utils.logger import get_logger

logger = get_logger()

def get_system_info() -> Dict[str, Any]:
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'python_version': sys.version,
        'system_memory_gb': None,
        'disk_space_gb': None,
        'platform': platform.system(),
        'platform_release': platform.release(),
        'processor': platform.processor(),
    }

    if torch.cuda.is_available():
        try:
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Detailed GPU info for first device
            compute_cap = torch.cuda.get_device_capability(0)
            info['gpu_compute_capability'] = f"{compute_cap[0]}.{compute_cap[1]}"
        except Exception as e:
            info['cuda_error'] = str(e)

    # System memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['system_memory_gb'] = memory.total / 1e9
        info['available_memory_gb'] = memory.available / 1e9
        info['memory_percent_used'] = memory.percent
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)

        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info['cpu_freq_current_mhz'] = cpu_freq.current
                info['cpu_freq_min_mhz'] = cpu_freq.min
                info['cpu_freq_max_mhz'] = cpu_freq.max
        except:
            pass

        # CPU usage
        try:
            info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        except:
            pass
    except ImportError:
        info['psutil_available'] = False

    # Disk space
    try:
        disk_usage = shutil.disk_usage(".")
        info['disk_space_gb'] = disk_usage.free / 1e9
        info['disk_total_gb'] = disk_usage.total / 1e9
        info['disk_used_gb'] = disk_usage.used / 1e9
        info['disk_percent_used'] = (disk_usage.used / disk_usage.total) * 100
    except Exception:
        pass

    return info

def print_system_diagnostics(system_info):
    logger.header("System Information")
    logger.metrics(names=['Python', 'PyTorch'], values=[system_info.get('python_version', 'Unknown')[:70], system_info.get('pytorch_version', 'Unknown')])
    if torch.cuda.is_available():
        logger.metrics(names=['CUDA', 'GPU Count'], values=[system_info.get('cuda_version', 'Unknown'), system_info.get('gpu_count', 0)])
        names = []
        values = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            compute_cap = torch.cuda.get_device_capability(i)
            names.append(f"GPU{i}")
            values.append(f"{props.name}, sm_{compute_cap[0]}.{compute_cap[1]}, {props.total_memory / 1e9:.2f}GB")
        logger.metrics(names=names, values=values)
    else:
        logger.warning("WARNING: CUDA not available, using CPU only.")

    if system_info.get('system_memory_gb'):
        total_mem = system_info.get('system_memory_gb', 0)
        avail_mem = system_info.get('available_memory_gb', 0)
        logger.metrics(names=['Available Memory', 'Total Memory'], values=[f"{avail_mem:.2f}", f"{total_mem:.2f}"], units=['GB', 'GB'])

    if system_info.get('cpu_count'):
        metric_names = ['CPU']
        metric_values = [str(system_info.get('cpu_count', 0))]
        metric_units = ['cores']
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metric_names.append('CPU Freq)')
                metric_values.append(f"{cpu_freq.current:.0f}")
                metric_units.append('MHz')
        except:
            pass

        logger.metrics(names=metric_names, values=metric_values, units=metric_units)

    if system_info.get('disk_space_gb'):
        disk_free = system_info.get('disk_space_gb', 0)
        try:
            disk_usage = shutil.disk_usage(".")
            disk_total = disk_usage.total / 1e9
            logger.metrics(names=['Disk Free', 'Disk Total'], values=[f"{disk_free:.2f}", f"{disk_total:.2f}"], units=['GB', 'GB'])
        except:
            logger.metric(f"Disk: {disk_free:.2f} GB free")
            logger.metrics(names=['Disk Free'], values=[f"{disk_free:.2f}"], units=['GB'])
