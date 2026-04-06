"""Kubernetes node discovery using the kubernetes client + dstack GPU helpers."""

from dataclasses import dataclass, field
from typing import Optional

from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class NodeInfo:
    name: str
    accelerator_type: Optional[str]
    accelerator_count: int = 0
    accelerator_available: int = 0
    cpu_count: Optional[float] = None
    memory_gb: Optional[float] = None
    is_ready: bool = True


@dataclass
class KubernetesNodesInfo:
    node_info_dict: dict[str, NodeInfo] = field(default_factory=dict)


k8_nodes: Optional[KubernetesNodesInfo] = None
context: Optional[str] = None


def init_kubernetes():
    """Discover K8s nodes and their GPU/CPU/memory resources.

    Uses the standard ``kubernetes`` client for node listing and
    dstack's label-based GPU detection helpers.
    """
    from kubernetes import client as k8s_client, config as k8s_config

    global k8_nodes, context

    try:
        k8s_config.load_incluster_config()
        context = "incluster"
    except k8s_config.ConfigException:
        contexts, active = k8s_config.list_kube_config_contexts()
        context = active["name"] if active else None
        k8s_config.load_kube_config(context=context)

    api = k8s_client.CoreV1Api()
    nodes = api.list_node().items

    # Compute GPUs in use by querying pod resource requests per node
    gpu_used_per_node: dict[str, int] = {}
    pods = api.list_pod_for_all_namespaces(field_selector="status.phase=Running").items
    for pod in pods:
        node_name = pod.spec.node_name if pod.spec else None
        if not node_name:
            continue
        for container in pod.spec.containers or []:
            requests = (container.resources.requests or {}) if container.resources else {}
            gpu_req = int(requests.get("nvidia.com/gpu", "0"))
            if gpu_req > 0:
                gpu_used_per_node[node_name] = gpu_used_per_node.get(node_name, 0) + gpu_req

    info_dict: dict[str, NodeInfo] = {}
    for node in nodes:
        name = node.metadata.name if node.metadata else "unknown"

        # Readiness
        is_ready = False
        for cond in (node.status.conditions or []) if node.status else []:
            if cond.type == "Ready":
                is_ready = cond.status == "True"
                break

        # Allocatable resources
        allocatable = node.status.allocatable or {} if node.status else {}
        cpu_str = allocatable.get("cpu", "0")
        mem_str = allocatable.get("memory", "0")
        gpu_count = int(allocatable.get("nvidia.com/gpu", "0"))

        cpu_count = _parse_cpu(cpu_str)
        memory_gb = _parse_memory_gb(mem_str)

        # GPU detection via dstack helpers (label-based)
        accelerator_type = None
        try:
            from dstack._internal.core.backends.kubernetes.resources import (
                get_node_labels, NVIDIA_GPU_PRODUCT_LABEL
            )
            labels = get_node_labels(node)
            gpu_product = labels.get(NVIDIA_GPU_PRODUCT_LABEL)
            if gpu_product is not None and "NVIDIA-" in gpu_product:
                gpu_product = gpu_product.split("NVIDIA-")[-1]
                accelerator_type = gpu_product
                if gpu_count == 0:
                    gpu_count = 1
        except Exception:
            # Fallback: check common label
            labels = (node.metadata.labels or {}) if node.metadata else {}
            accelerator_type = labels.get(
                "nvidia.com/gpu.product",
                labels.get("accelerator", None),
            )

        gpu_used = gpu_used_per_node.get(name, 0)

        info_dict[name] = NodeInfo(
            name=name,
            accelerator_type=accelerator_type,
            accelerator_count=gpu_count,
            accelerator_available=max(gpu_count - gpu_used, 0),
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            is_ready=is_ready,
        )

    k8_nodes = KubernetesNodesInfo(node_info_dict=info_dict)
    if not info_dict:
        raise RuntimeError("Failed to load Kubernetes node info on startup.")


def _parse_cpu(val: str) -> float:
    """Parse K8s CPU quantity (e.g. ``'4'``, ``'500m'``) into float cores."""
    if val.endswith("m"):
        return int(val[:-1]) / 1000.0
    return float(val)


def _parse_memory_gb(val: str) -> float:
    """Parse K8s memory quantity (e.g. ``'16Gi'``, ``'16384Mi'``) into GB."""
    val = val.strip()
    if val.endswith("Ki"):
        return int(val[:-2]) / (1024 * 1024)
    if val.endswith("Mi"):
        return int(val[:-2]) / 1024
    if val.endswith("Gi"):
        return int(val[:-2])
    if val.endswith("Ti"):
        return int(val[:-2]) * 1024
    # Plain bytes
    try:
        return int(val) / (1024 ** 3)
    except ValueError:
        return 0.0
