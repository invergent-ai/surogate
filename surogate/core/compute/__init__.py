"""Compute module — SkyPilot initialization and monkeypatches.

SkyPilot is used as a direct library (bypassing its REST API server).
We call the implementation layer: sky.execution, sky.core, sky.jobs.server.core.

Monkeypatches are applied before any sky imports to support
platform-specific features.
"""

import os
import uuid

from typing import Optional

from surogate.utils.logger import get_logger

logger = get_logger()

_initialized = False

dummy_request_id = str(uuid.uuid4())  # Used in skylet for non-SkyPilot requests

def init_skypilot():
    """Initialize SkyPilot as a library.

    1. Initialize global user state (SkyPilot's SQLite DB).
    2. Reload SkyPilot config.
    """
    
    global _initialized
    if _initialized:
        return

    os.environ.setdefault("ENV_VAR_IS_SKYPILOT_SERVER", "1")
        
    # ── Patch get_current_request_id to return a real UUID ──────
    from sky.utils import common_utils
    _original_get_request_id = common_utils.get_current_request_id

    def _patched_get_current_request_id() -> str:
        value = _original_get_request_id()
        if value == "dummy-request-id":
            return dummy_request_id
        return value

    common_utils.get_current_request_id = _patched_get_current_request_id
    
    # ── Patch skypilot to work with our projects ──────
    import sky.check
    import sky.skypilot_config as skypilot_config
    import sky.data.storage as sky_storage
    import sky.global_user_state as global_user_state
    import surogate.core.compute.skypilot.patcher as patcher
    
    sky.check._get_workspace_allowed_clouds = patcher._surogate_get_workspace_allowed_clouds
    skypilot_config.get_active_workspace = patcher._surogate_get_active_workspace
    sky_storage.get_cached_enabled_storage_cloud_names_or_refresh = patcher._surogate_get_cached_enabled_storage_cloud_names_or_refresh
    global_user_state.get_cached_enabled_clouds = patcher._surogate_get_cached_enabled_clouds
    global_user_state.get_allowed_clouds = patcher._surogate_get_allowed_clouds

    # ── Remove artificial service limits ─────────────────────────
    patch_serve_limits()

    # ── SkyPilot state initialization ────────────────────────────
    set_consolidattion_mode()
    set_image_pull_policy_if_not_present()
    
    from sky import global_user_state
    from sky import skypilot_config

    global_user_state.initialize_and_get_db()
    skypilot_config.safe_reload_config()

    _initialized = True
    logger.info("SkyPilot initialized (direct library mode)")

    from surogate.core.compute.kubernetes import init_kubernetes
    init_kubernetes()
    

def set_image_pull_policy_if_not_present():
    # ── Patch kubernetes-ray template: Always → IfNotPresent ────
    from pathlib import Path
    import sky
    _tpl = Path(sky.__file__).parent / 'templates' / 'kubernetes-ray.yml.j2'
    _content = _tpl.read_text()
    _patched = _content.replace('imagePullPolicy: Always', 'imagePullPolicy: IfNotPresent')
    if _patched != _content:
        _tpl.write_text(_patched)
        logger.info("Patched kubernetes-ray.yml.j2: imagePullPolicy → IfNotPresent")
        

def patch_serve_limits():
    """Widen SkyPilot serve limits and replace the per-service load balancer.

    By default SkyPilot only opens ports 30001-30020 (20 services) and
    caps the number of services based on available memory.  Since we run
    in consolidation mode the port range firewall rule is irrelevant and
    we manage our own resource budget, so we remove both limits.

    We also replace the load-balancer subprocess with a no-op: instead of
    spawning a separate uvicorn process per service, all proxying is
    handled by the Surogate server's own proxy route
    (``/api/models/serve/{service_name}/...``).  The SkyPilot controller
    process still runs and manages replicas / autoscaling — we just talk
    to it from our proxy sync loop.
    """
    from sky.serve import constants as serve_constants
    from sky.serve import load_balancer
    from sky.utils import controller_utils

    serve_constants.LOAD_BALANCER_PORT_RANGE = '30001-31000'
    controller_utils.can_start_new_process = lambda pool: True

    def _noop_load_balancer(*args, **kwargs):
        """Replaces SkyPilot's per-service LB process.

        The actual proxying is handled by the Surogate server.
        Registration with our proxy happens in the monitor when
        the controller port becomes known.
        """
        pass

    load_balancer.run_load_balancer = _noop_load_balancer


def set_consolidattion_mode():
    # ── Ensure ~/.sky/config.yaml exists with consolidation_mode ──
    from pathlib import Path
    _sky_config = Path.home() / '.sky' / 'config.yaml'
    if not _sky_config.exists():
        _sky_config.parent.mkdir(parents=True, exist_ok=True)
        _sky_config.write_text(
            'jobs:\n'
            '  controller:\n'
            '    consolidation_mode: true\n'
            'serve:\n'
            '  controller:\n'
            '    consolidation_mode: true\n'
        )
