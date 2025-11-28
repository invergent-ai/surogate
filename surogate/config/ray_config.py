from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Dict, List, Optional
from typing_extensions import Literal
from surogate.utils.dict import DictDefault

@dataclass
class RayDeviceGroup:
    """
    Ray device group configuration.
    Args:
        device (str): Device type, currently supports GPU/CPU
        ranks (str): Which ranks are allocated to the current group. For device=CPU, ranks can only be an integer representing the total number of processes needed. If GPU, can be in formats like `[0,1,2,3]`, `4`, `list(range(0, 4))`, etc.
        workers (List[str]): Roles are allocated to the current group: 'default', 'prm', 'orm', 'sampler'
    """
    device: Optional[str] = None
    ranks: Optional[str] = None
    workers: Optional[List[Literal['default', 'prm', 'orm', 'sampler']]] = None

    def __init__(self, cfg: Dict[str, Any]):
        self.device = cfg['device']
        self.ranks = cfg['ranks']
        self.workers = cfg['workers']


@dataclass
class RayConfig(ABC):
    """
    Ray configuration for distributed training.

    Args:
        nproc_per_node (int): Required number of GPUs per node
        use_ray (bool): Whether to use Ray for distributed training
    """
    ray_nproc_per_node: Optional[int] = None
    use_ray: Optional[bool] = False
    ray_groups: Optional[Dict[str, RayDeviceGroup]] = None
    
    def __init__(self, cfg: DictDefault):
        self.use_ray = cfg.get('use_ray', False)
        if not self.use_ray:
            return
        
        if 'device_groups' not in cfg:
            raise ValueError('device_groups are required when use_ray is True.')
        
        device_groups = cfg['device_groups']
        self.ray_nproc_per_node = device_groups.get('nproc_per_node', None)
        self.ray_groups = {}
        for name in device_groups:
            if name not in ['nproc_per_node']:
                self.ray_groups[name] = RayDeviceGroup(device_groups[name])

        
