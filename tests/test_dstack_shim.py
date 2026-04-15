"""End-to-end smoke test for the dstack pydantic.v1 shim.

Verifies that with pydantic v2 installed process-wide, dstack still boots
under its original pydantic v1 API thanks to the source-rewriting loader
in `surogate/__init__.py`.

Covers:
  * Alembic migrations run (SourceFileLoader resource reader still works)
  * Admin user + default project persist and round-trip
  * dstack core pydantic models (RunSpec) build and remain v1 subclasses
  * APScheduler background tasks start and shut down cleanly
  * surogate's own models are real pydantic v2 in the same process

Run: `uv run pytest tests/test_dstack_shim.py -s`
"""

import asyncio
import os
import shutil
import tempfile

import pytest


@pytest.mark.asyncio
async def test_dstack_boots_with_pydantic_v2_in_process():
    tmp = tempfile.mkdtemp(prefix="dstack-shim-test-")
    os.makedirs(os.path.join(tmp, "data"), exist_ok=True)
    os.environ["DSTACK_SERVER_DIR"] = tmp
    os.environ["DSTACK_DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/test.db"

    try:
        import surogate  # noqa: F401 — installs the rewriter
        from surogate.core.compute import (
            init_dstack,
            get_dstack_admin,
            ensure_dstack_project,
            shutdown_dstack,
        )

        await init_dstack()

        admin = get_dstack_admin()
        assert admin.name == "admin"
        assert str(admin.global_role) in ("GlobalRole.ADMIN", "admin")

        from dstack._internal.core.models.configurations import TaskConfiguration
        from dstack._internal.core.models.profiles import Profile
        from dstack._internal.core.models.runs import RunSpec
        from pydantic.v1 import BaseModel as V1Base
        from pydantic import BaseModel as V2Base

        cfg = TaskConfiguration(type="task", image="python:3.11", commands=["echo hi"])
        spec = RunSpec(configuration=cfg, profile=Profile(name="default"))
        assert isinstance(spec, V1Base), "dstack RunSpec must be pydantic v1"
        assert not isinstance(spec, V2Base), "dstack RunSpec must NOT be pydantic v2"
        dumped = spec.dict()
        assert "run_name" in dumped and "repo_id" in dumped

        proj = await ensure_dstack_project("shimtest")
        assert proj.name == "shimtest"
        assert proj.id is not None

        from surogate.server.models.compute import JobResponse
        assert issubclass(JobResponse, V2Base), "surogate models must be pydantic v2"
        assert not issubclass(JobResponse, V1Base)

        await shutdown_dstack()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(test_dstack_boots_with_pydantic_v2_in_process())
    print("OK")
