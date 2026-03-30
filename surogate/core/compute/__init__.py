"""Compute module — SkyPilot initialization and monkeypatches.

SkyPilot is used as a direct library (bypassing its REST API server).
We call the implementation layer: sky.execution, sky.core, sky.jobs.server.core.

Monkeypatches are applied before any sky imports to support
platform-specific features.
"""

import os
from pathlib import Path

from surogate.utils.logger import get_logger

logger = get_logger()

_initialized = False

# All SkyPilot artifacts live under ~/.surogate/.sky instead of ~/.sky.
#
# SkyPilot resolves dynamic paths via:
#   runtime_utils.get_runtime_dir_path('.sky/foo')
#     → os.path.join(os.environ.get('SKY_RUNTIME_DIR', '~'), '.sky/foo')
#
# Setting SKY_RUNTIME_DIR=~/.surogate makes everything land in
# ~/.surogate/.sky/ (state.db, locks, logs, etc.).
# Hardcoded '~/.sky/...' string constants are patched to match.
SUROGATE_HOME = Path.home() / ".surogate"
SUROGATE_SKY_DIR = SUROGATE_HOME / ".sky"


def _redirect_sky_home():
    """Set SKY_RUNTIME_DIR so all dynamic paths resolve under ~/.surogate/.sky."""
    os.environ.setdefault("SKY_RUNTIME_DIR", str(SUROGATE_HOME))
    SUROGATE_SKY_DIR.mkdir(parents=True, exist_ok=True)


def _patch_hardcoded_paths():
    """Patch constants that use literal '~/.sky' strings."""
    import sky.skypilot_config as cfg

    sky_dir = str(SUROGATE_SKY_DIR)
    cfg._GLOBAL_CONFIG_PATH = f"{sky_dir}/config.yaml"
    cfg.SKYPILOT_CONFIG_LOCK_PATH = f"{sky_dir}/locks/.skypilot_config.lock"


def init_skypilot():
    """Initialize SkyPilot as a library.

    1. Redirect ~/.sky → ~/.surogate/sky (env var + constant patches).
    2. Initialize global user state (SkyPilot's SQLite DB).
    3. Reload SkyPilot config.
    """
    global _initialized
    if _initialized:
        return

    # ── Redirect paths (before heavy sky imports) ────────────────
    _redirect_sky_home()

    # ── Additional monkeypatches ─────────────────────────────────
    # Apply further patches here before initializing state.

    # ── Patch hardcoded path constants ───────────────────────────
    _patch_hardcoded_paths()

    # ── SkyPilot state initialization ────────────────────────────
    from sky import global_user_state
    from sky import skypilot_config

    global_user_state.initialize_and_get_db()
    skypilot_config.safe_reload_config()

    _initialized = True
    logger.info(
        "SkyPilot initialized (direct library mode, home=%s)", SUROGATE_SKY_DIR
    )
