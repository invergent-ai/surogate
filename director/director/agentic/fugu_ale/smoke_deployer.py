"""Zero-paid real ALE transport/artifact/grader smoke for the Amber train task."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import ClassVar

from ale_run.base_interface import AgentRunResult, BaseAgentDeployer, TrajectoryBuilder

from .config import AmberSmokeConfig

_REPORT_NAME = "amber_zero_paid_smoke.json"
_EXPECTED_INPUTS = (
    "complex_structure.pdb",
    "input_environment_spec.md",
    "task_sop.md",
)


def amber_positive_outputs() -> dict[str, str]:
    """Return a deterministic positive fixture; never use it as training data."""
    basename = "GLN_phb2_lc3_aurka_model_0"
    return {
        "leap.in": f"""source leaprc.protein.ff14SB
set default PBRadii mbondi3
complex = loadpdb complex_structure.pdb
saveamberparm complex {basename}.prmtop {basename}.inpcrd
savepdb complex {basename}_fixed.pdb
quit
""",
        "step2_implicit.mini.mdin": """Implicit-solvent minimization
&cntrl
  imin=1, ntb=0, cut=999.0, igb=8,
  maxcyc=5000, ncyc=2500, ntpr=100, ntxo=2,
  saltcon=0.1, intdiel=1.0, extdiel=80.0,
/
""",
        "submit_min.sh": f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
set -euo pipefail
module load amber/22
module load cuda/11.6.2
BASE={basename}
mkdir -p params
if [ ! -f params/${{BASE}}.prmtop ] || [ ! -f params/${{BASE}}.inpcrd ]; then
  tleap -f leap.in
  mv "${{BASE}}.prmtop" "${{BASE}}.inpcrd" "${{BASE}}_fixed.pdb" params/
fi
$AMBERHOME/bin/pmemd.cuda -O -i step2_implicit.mini.mdin -p params/${{BASE}}.prmtop -c params/${{BASE}}.inpcrd -ref params/${{BASE}}.inpcrd -o min.out -r min.rst
""",
    }


def _prompt_path(prompt: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*\n- `([^`]+)`", prompt)
    if match is None:
        raise ValueError(f"Amber smoke prompt lacks {label!r}")
    return match.group(1)


class AmberSmokeDeployer(BaseAgentDeployer):
    """Write a known positive fixture through the real remote ALE session."""

    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (_REPORT_NAME,)

    @property
    def version(self) -> str | None:
        return "20260721-amber-zero-paid-smoke-v2"

    async def install(self) -> None:
        from cua_bench.computers.remote import RemoteDesktopSession  # noqa: F401

        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)
        if not self.executor.sandbox.is_linux:
            raise RuntimeError("Amber smoke requires the Linux ALE image")

    async def launch(self, prompt: str) -> AgentRunResult:
        from cua_bench.computers.remote import RemoteDesktopSession

        started = time.monotonic()
        if (
            "complex_structure.pdb" not in prompt
            or "GLN_phb2_lc3_aurka_model_0" not in prompt
        ):
            raise RuntimeError("Amber smoke refuses to run on any other task")
        input_dir = _prompt_path(prompt, "Input directory")
        output_match = re.search(
            r"Create exactly these three files under `([^`]+)`", prompt
        )
        if output_match is None:
            raise RuntimeError("Amber smoke cannot locate the output directory")
        output_dir = output_match.group(1)
        session = RemoteDesktopSession(
            api_url=self.executor.sandbox.endpoint,
            os_type=self.executor.sandbox.os,
            ephemeral=False,
            headless=True,
        )
        ready = await session.wait_until_ready(timeout=self.config.connect_timeout_s)
        if not ready:
            raise RuntimeError("ALE remote desktop session did not become ready")
        missing = [
            name
            for name in _EXPECTED_INPUTS
            if not await session.file_exists(f"{input_dir}/{name}")
        ]
        if missing:
            raise RuntimeError(f"ALE did not stage the authentic Amber inputs: {missing}")
        await session.run_command(f"mkdir -p '{output_dir}'", check=True)
        existing = [
            entry
            for entry in await session.list_dir(output_dir)
            if Path(str(entry)).name != ".gitkeep"
        ]
        if existing:
            raise RuntimeError("Amber smoke output directory was not clean")
        outputs = amber_positive_outputs()
        for name, content in outputs.items():
            await session.write_file(f"{output_dir}/{name}", content)
        report = {
            "version": self.version,
            "verdict": "AMBER_ZERO_PAID_REMOTE_FIXTURE_WRITTEN",
            "input_dir": input_dir,
            "output_dir": output_dir,
            "input_files": list(_EXPECTED_INPUTS),
            "output_files": sorted(outputs),
            "external_calls": 0,
            "paid_calls": 0,
            "optimizer_steps": 0,
            "training_eligible": False,
        }
        report_path = Path(self.executor.work_dir) / _REPORT_NAME
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return AgentRunResult(
            status="completed",
            transcript_path=str(report_path),
            duration_s=time.monotonic() - started,
        )

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: AmberSmokeConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        del config
        path = work_dir / _REPORT_NAME
        if not path.is_file():
            builder.add_step(
                source="system",
                message="Amber zero-paid smoke report is missing.",
                extra={"run_status": run_result.status},
            )
            return
        report = json.loads(path.read_text(encoding="utf-8"))
        builder.add_step(
            source="system",
            message="Amber zero-paid remote fixture was written for grader smoke.",
            extra=report,
        )
        builder.trajectory.extra["amber_zero_paid_smoke"] = report
