import contextlib
import functools
import pathlib
import json
import sys
from typing import List, Dict, Union, Callable, Any

from surogate import _surogate

from surogate.core.config.sft_config import SFTConfig


@contextlib.contextmanager
def training_logger_context(config: SFTConfig):
    report_to = config.report_to
    if report_to is None:
        backends: List[str] = []
    elif isinstance(report_to, str):
        backends = [report_to]
    else:
        backends = list(report_to)
    backends_set = set(backends)

    # Prepare options once (used for TrainingRunLogger + optional backend init).
    # NOTE: Avoid `dataclasses.asdict` here: it deep-copies fields and can choke on
    # nanobind/extension objects (e.g., RuntimeOptions).
    log_options = dict(vars(config))
    log_options.pop("model_info")
    log_options.pop("model_template")
    log_options.pop("model")
    log_options.pop("tokenizer")
    
    filtered_options: Dict[str, Union[bool, int, float, str]] = {}
    for k, v in log_options.items():
        if v is None:
            filtered_options[k] = ""
        elif isinstance(v, bool):
            filtered_options[k] = v
        elif isinstance(v, int):
            filtered_options[k] = v
        elif isinstance(v, float):
            filtered_options[k] = v
        elif isinstance(v, str):
            filtered_options[k] = v
        else:
            filtered_options[k] = str(v)

    with contextlib.ExitStack() as stack:
        handlers: List[Callable[[dict], None]] = []

        if "wandb" in backends_set:
            import wandb

            project = getattr(config, "wandb_project", None) or "Surogate"
            name = getattr(config, "wandb_name", None) or getattr(config, "run_name", None)

            # When resuming from checkpoint, try to resume the existing wandb run
            resume_mode = "allow"  # Default: create new or resume if id matches
            run_id = getattr(config, "wandb_id", None)

            checkpoint_dir = getattr(config, "checkpoint_dir", None)
            resume_from_checkpoint = getattr(config, "resume_from_checkpoint", False)

            if resume_from_checkpoint and checkpoint_dir and not run_id:
                # Try to load run_id from checkpoint metadata if available
                checkpoint_path = pathlib.Path(checkpoint_dir)
                if (checkpoint_path / "wandb_id.txt").exists():
                    try:
                        run_id = (checkpoint_path / "wandb_id.txt").read_text().strip()
                        resume_mode = "must"  # Force resume if we have a stored id
                    except Exception:
                        pass

            wandb_run = stack.enter_context(
                wandb.init(
                    project=project,
                    name=name,
                    id=run_id,
                    resume=resume_mode,
                    config=filtered_options,
                )
            )

            # Save the run id for future resumption
            if checkpoint_dir:
                try:
                    checkpoint_path = pathlib.Path(checkpoint_dir)
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    (checkpoint_path / "wandb_id.txt").write_text(wandb_run.id)
                except Exception:
                    pass

            handlers.append(functools.partial(log_line_to_wandb, wandb_run))

        if "surogate" in backends_set:
            from surogate.train.metrics_writer import MetricsWriter
            
            metrics_path = getattr(config, "surogate_metrics_path", None)
            writer = stack.enter_context(MetricsWriter(output_path=metrics_path))
            handlers.append(functools.partial(log_line_to_surogate, writer))

        if "aim" in backends_set:
            aim_run = stack.enter_context(_aim_run_context(config))
            try:
                aim_run["hparams"] = filtered_options
            except Exception:
                pass
            handlers.append(functools.partial(log_line_to_aim, aim_run))

        log_callback = make_multi_backend_log_callback(handlers)

        train_logger = _surogate.TrainingRunLogger(
            str(config.log_file),
            callback=log_callback,
            verbosity=_surogate.LogVerbosity.DEFAULT,
        )
        train_logger.log_cmd(sys.argv)
        train_logger.log_options(filtered_options)
        yield train_logger


def log_line_to_wandb(run: "wandb.Run", entry: dict):
    kind = entry["log"]
    step = entry["step"]
    if kind == "step":
        step_tokens = entry.get("step_tokens", 0)
        duration_ms = entry.get("duration_ms", 0)
        tps = step_tokens / (duration_ms / 1000) if duration_ms else 0.0
        run.log(
            {
                f"train/{k}": v
                for k, v in entry.items()
                if k not in {"log", "step", "time", "step_tokens"}
            },
            step=step,
        )
        run.log({"train/tokens_per_second": tps}, step=step)
    elif kind == "eval":
        eval_tokens = entry.get("eval_tokens", 0)
        duration_ms = entry.get("duration_ms", 0)
        tps = eval_tokens / (duration_ms / 1000) if duration_ms else 0.0
        run.log(
            {
                f"eval/{k}": v
                for k, v in entry.items()
                if k not in {"log", "step", "time", "eval_tokens"}
            },
            step=step,
        )
        run.log({"eval/tokens_per_second": tps}, step=step)
    elif kind == "gpu":
        gpu_entry = {
            k: v
            for k, v in entry.items()
            if k not in {"log", "step", "time", "throttle", "id"}
        }
        if gpu_entry.get("fan", 0) == 0:
            gpu_entry.pop("fan", None)
        if "dram_free" in gpu_entry:
            gpu_entry["dram_free"] /= 1024 ** 2  # MiB
        if "pcie_rx" in gpu_entry:
            gpu_entry["pcie_rx"] /= 1024 ** 2  # MiB/s
        if "pcie_tx" in gpu_entry:
            gpu_entry["pcie_tx"] /= 1024 ** 2  # MiB/s
        run.log({f"gpu/{k}": v for k, v in gpu_entry.items()}, step=step)
    elif kind == "cmd":
        # where is belongs
        run.config["cmd"] = entry["cmd"]
    elif kind == "gpu-model":
        if entry["rank"] == 0:
            run.config["gpu"] = entry
        else:
            run.config[f"gpu-{entry['rank']}"] = entry
    elif kind == "allocator":
        import plotly.express as px
        names = [alloc["name"] for alloc in entry["stats"]]
        amounts = [round(alloc["device"] / 1024 / 1024, 1) for alloc in entry["stats"]]

        fig = px.pie(
            names=names,
            values=amounts,
            title=f"GPU Allocations",
        )
        run.log({"allocations": fig}, step=step)
    elif kind == "dataset":
        pass
        # run.config["dataset"] = entry
    elif kind in ["option", "info"]:
        pass
    elif kind == "message":
        print(entry["message"])
    elif kind == "sol":
        if entry["rank"] != 0:
            return
        import plotly.express as px
        names = ["Blocks", "LM-Head", "Attention"]
        amounts = [entry["blocks"], entry["lm_head"], entry["attention"]]

        fig = px.pie(
            names=names,
            values=amounts,
            title=f"FLOPs",
        )
        run.log({"ops": fig}, step=step)
    else:
        raise RuntimeError(f"Unknown kind {kind}")


def make_wandb_log_callback(run):
    def callback(entry: str):
        log_line_to_wandb(run, json.loads(entry))

    return callback


def log_line_to_aim(run: "aim.Run", entry: dict):
    kind = entry["log"]
    step = entry["step"]

    if kind == "step":
        step_tokens = entry.get("step_tokens", 0)
        duration_ms = entry.get("duration_ms", 0)
        tps = step_tokens / (duration_ms / 1000) if duration_ms else 0.0
        for k, v in entry.items():
            if k in {"log", "step", "time", "step_tokens"}:
                continue
            if not isinstance(v, (int, float)):
                continue
            run.track(v, name=f"train/{k}", step=step)
        run.track(tps, name="train/tokens_per_second", step=step)
    elif kind == "eval":
        eval_tokens = entry.get("eval_tokens", 0)
        duration_ms = entry.get("duration_ms", 0)
        tps = eval_tokens / (duration_ms / 1000) if duration_ms else 0.0
        for k, v in entry.items():
            if k in {"log", "step", "time", "eval_tokens"}:
                continue
            if not isinstance(v, (int, float)):
                continue
            run.track(v, name=f"eval/{k}", step=step)
        run.track(tps, name="eval/tokens_per_second", step=step)
    elif kind == "gpu":
        gpu_id = entry.get("id", 0)
        for k, v in entry.items():
            if k in {"log", "step", "time", "throttle", "id"}:
                continue
            if k == "fan" and v == 0:
                continue
            if k == "dram_free":
                v = v / 1024 ** 2  # MiB
            elif k in {"pcie_rx", "pcie_tx"}:
                v = v / 1024 ** 2  # MiB/s
            run.track(v, name=f"gpu/{gpu_id}/{k}", step=step)
    elif kind == "cmd":
        try:
            run["cmd"] = entry["cmd"]
        except Exception:
            pass
    elif kind == "gpu-model":
        try:
            if entry.get("rank", 0) == 0:
                run["gpu"] = entry
            else:
                run[f"gpu-{entry['rank']}"] = entry
        except Exception:
            pass
    elif kind == "allocator":
        # Prefer raw numeric logging (faster, backend-agnostic) over rich plots.
        stats = entry.get("stats") or []
        for alloc in stats:
            name = alloc.get("name", "unknown")
            device_mib = alloc.get("device", 0) / 1024 ** 2
            run.track(device_mib, name=f"allocator/{name}_mib", step=step)
    elif kind == "dataset":
        pass
    elif kind in ["option", "info"]:
        pass
    elif kind == "message":
        print(entry.get("message", ""))
    elif kind == "sol":
        if entry.get("rank", 0) != 0:
            return
        run.track(entry.get("blocks", 0), name="flops/blocks", step=step)
        run.track(entry.get("lm_head", 0), name="flops/lm_head", step=step)
        run.track(entry.get("attention", 0), name="flops/attention", step=step)
    else:
        raise RuntimeError(f"Unknown kind {kind}")


def log_line_to_surogate(writer, entry: dict):
    kind = entry["log"]
    step = entry["step"]

    if kind == "step":
        metrics = {
            f"train/{k}": v for k, v in entry.items()
            if k not in {"log", "step", "time", "step_tokens"} and isinstance(v, (int, float))
        }
        step_tokens = entry.get("step_tokens", 0)
        duration_ms = entry.get("duration_ms", 0)
        if duration_ms:
            metrics["train/tokens_per_second"] = step_tokens / (duration_ms / 1000)
        writer.track(step, **metrics)
    elif kind == "eval":
        metrics = {
            f"eval/{k}": v for k, v in entry.items()
            if k not in {"log", "step", "time", "eval_tokens"} and isinstance(v, (int, float))
        }
        eval_tokens = entry.get("eval_tokens", 0)
        duration_ms = entry.get("duration_ms", 0)
        if duration_ms:
            metrics["eval/tokens_per_second"] = eval_tokens / (duration_ms / 1000)
        writer.track(step, **metrics)
    elif kind in {"gpu", "cmd", "gpu-model", "allocator", "dataset",
                   "option", "info", "message", "sol"}:
        pass  # skip non-training metrics
    else:
        raise RuntimeError(f"Unknown kind {kind}")


def make_multi_backend_log_callback(handlers: List[Callable[[dict], None]]):
    if not handlers:
        return None

    if len(handlers) == 1:
        handler = handlers[0]

        def callback(entry: str):
            handler(json.loads(entry))

        return callback

    def callback(entry: str):
        parsed = json.loads(entry)
        for handler in handlers:
            handler(parsed)

    return callback


@contextlib.contextmanager
def _aim_run_context(config: SFTConfig):
    import aim

    experiment = getattr(config, "aim_experiment", None) or "Surogate"
    repo = getattr(config, "aim_repo", None)
    run_name = getattr(config, "aim_name", None) or getattr(config, "run_name", None)

    kwargs: Dict[str, Any] = {"experiment": experiment}
    if repo:
        kwargs["repo"] = repo

    # When resuming from checkpoint, try to resume the existing run using stored hash
    checkpoint_dir = getattr(config, "checkpoint_dir", None)
    resume_from_checkpoint = getattr(config, "resume_from_checkpoint", False)
    
    run_hash = None
    if resume_from_checkpoint and checkpoint_dir:
        # Try to load run_hash from checkpoint metadata if available
        checkpoint_path = pathlib.Path(checkpoint_dir)
        if (checkpoint_path / "aim_run_hash.txt").exists():
            try:
                run_hash = (checkpoint_path / "aim_run_hash.txt").read_text().strip()
            except Exception:
                pass

    if run_hash:
        kwargs["run_hash"] = run_hash

    run = aim.Run(**kwargs)
    try:
        try:
            run.name = run_name
        except Exception:
            pass

        # Save the run hash for future resumption
        if checkpoint_dir:
            try:
                checkpoint_path = pathlib.Path(checkpoint_dir)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                (checkpoint_path / "aim_run_hash.txt").write_text(run.hash)
            except Exception:
                pass

        yield run
    finally:
        close = getattr(run, "close", None)
        if callable(close):
            close()