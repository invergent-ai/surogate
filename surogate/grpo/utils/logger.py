import atexit
import json as json_module
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

# Global logger instance
_LOGGER = None
_JSON_LOGGING = False

NO_BOLD = "\033[22m"
RESET = "\033[0m"


def build_log_entry(record) -> dict:
    """Build a flat JSON log entry from a loguru record."""
    extra = record["extra"]

    # Handle progress events specially - emit structured progress format
    if extra.get("_progress"):
        return {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "type": "progress",
            "desc": extra["desc"],
            "current": extra["current"],
            "total": extra["total"],
            "percent": extra["percent"],
            **({"step": extra["step"]} if extra.get("step") is not None else {}),
            **({"extra": extra["postfix"]} if extra.get("postfix") else {}),
        }

    # Standard log entry
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    if record["exception"] is not None:
        exc = record["exception"]
        log_entry["exception"] = "".join(traceback.format_exception(exc.type, exc.value, exc.traceback))
    # Extract tag from extra if present (used by workers to identify themselves)
    if extra:
        if "tag" in extra:
            log_entry["tag"] = extra["tag"]
            extra = {k: v for k, v in extra.items() if k != "tag"}
        if extra:
            log_entry["extra"] = extra
    return log_entry


def json_sink(message) -> None:
    """Sink that outputs flat JSON to stdout for log aggregation (Loki, Grafana, etc.)."""
    log_entry = build_log_entry(message.record)
    sys.stdout.write(json_module.dumps(log_entry) + "\n")
    sys.stdout.flush()


class JsonFileSink:
    """File sink that keeps the handle open and writes flat JSON lines."""

    def __init__(self, log_file: Path):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_file, "a")
        atexit.register(self._close)

    def _close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def write(self, message) -> None:
        log_entry = build_log_entry(message.record)
        self._file.write(json_module.dumps(log_entry) + "\n")
        self._file.flush()


class InterceptHandler(logging.Handler):
    """Intercept standard logging library and routes to our prime-rl logger with specified prefix."""

    def __init__(self, prefix: str | None):
        super().__init__()
        self.prefix = prefix

    def emit(self, record: logging.LogRecord) -> None:
        logger = get_logger()
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        message = record.getMessage()
        if self.prefix is not None:
            message = f"[{self.prefix}] {message}"
        logger.opt(depth=depth, exception=record.exc_info).log(level, message)


def setup_logger(
    log_level: str = "info",
    log_file: Path | None = None,
    append: bool = False,
    tag: str | None = None,
    json_logging: bool = False,
):
    global _LOGGER, _JSON_LOGGING
    _JSON_LOGGING = json_logging

    # Clean up old logger instance to prevent resource leaks
    if _LOGGER is not None:
        _LOGGER.remove()

    # Format message with optional tag prefix
    tag_prefix = f"[{tag}] " if tag else ""
    message = "".join(
        [
            " <level>{level: >7}</level>",
            f" <level>{NO_BOLD}",
            f"{tag_prefix}{{message}}",
            f"{RESET}</level>",
        ]
    )
    time = "<dim>{time:HH:mm:ss}</dim>"
    if log_level.upper() != "DEBUG":
        debug = ""
    else:
        debug = "".join([f"<level>{NO_BOLD}", " [{file}::{line}]", f"{RESET}</level>"])
    format = time + message + debug

    # NOTE: We are creating a new "module-level" logger instance for prime-rl so that third-party code cannot "hijack" our logger
    # This is a bit hacky because loguru does not publicly expose the logger class, but oh well, it works
    from loguru._logger import Core as _Core
    from loguru._logger import Logger as _Logger

    logger = _Logger(
        core=_Core(),
        exception=None,
        depth=0,
        record=False,
        lazy=False,
        colors=False,
        raw=False,
        capture=True,
        patchers=[],
        extra={},
    )

    # Bind tag to logger context for JSON mode (in non-JSON mode, tag is in the format string)
    if json_logging and tag:
        logger = logger.bind(tag=tag)

    # Install console handler (enqueue=True only for JSON mode to avoid blocking in async contexts)
    if json_logging:
        logger.add(json_sink, level=log_level.upper(), enqueue=True)
    else:
        logger.add(sys.stdout, format=format, level=log_level.upper(), colorize=True)

    # If specified, install file handler
    if log_file is not None:
        if not append and log_file.exists():
            log_file.unlink()
        if json_logging:
            file_sink = JsonFileSink(log_file)
            logger.add(file_sink.write, level=log_level.upper(), enqueue=True)
        else:
            logger.add(log_file, format=format, level=log_level.upper(), colorize=True)

    # Disable critical logging
    logger.critical = lambda _: None

    # Set the global logger instance
    _LOGGER = logger

    return logger


def get_logger():
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = setup_logger()
    return _LOGGER


def reset_logger():
    """Reset the logger. Useful mainly in tests."""
    global _LOGGER, _JSON_LOGGING
    if _LOGGER is not None:
        _LOGGER.remove()
    _LOGGER = None
    _JSON_LOGGING = False


class ProgressTracker:
    """Progress tracker that uses tqdm or logs progress when JSON logging is enabled."""

    def __init__(
        self,
        total: int,
        desc: str,
        json_logging: bool | None = None,
        log_every_percent: int = 10,
        step: int | None = None,
    ):
        self.total = total
        self.desc = desc
        self.step = step
        self.json_logging = json_logging if json_logging is not None else _JSON_LOGGING
        self.log_every_percent = log_every_percent
        self.current = 0
        self._last_logged_percent = -log_every_percent
        self._postfix: dict[str, Any] = {}
        # Plain-log progress cadence (2026-07-29): tqdm renders \r to a TTY,
        # so under nohup/file logging batch progress was INVISIBLE — the log
        # showed per-env stats but never how far the current step was. Emit
        # an INFO line with rate + ETA every log_every_percent OR every
        # `heartbeat_s`, whichever comes first.
        import time as _time

        self._time = _time
        self._start_time = _time.time()
        self._last_line_time = 0.0
        self._heartbeat_s = 60.0

        if self.json_logging:
            self._pbar = None
            # Don't log 0% on init - only log on actual progress
        else:
            from tqdm import tqdm

            self._pbar = tqdm(total=total, desc=desc)

    def update(self, n: int = 1):
        self.current += n
        if self._pbar is not None:
            self._pbar.update(n)
            self._maybe_log_line()
        else:
            self._log_progress()

    def _maybe_log_line(self):
        """Plain INFO progress line for file/nohup logs (non-JSON mode)."""
        percent = int(100 * self.current / self.total) if self.total > 0 else 0
        now = self._time.time()
        due_percent = percent >= self._last_logged_percent + self.log_every_percent
        due_time = now - self._last_line_time >= self._heartbeat_s
        if not (due_percent or due_time or self.current >= self.total):
            return
        elapsed = max(now - self._start_time, 1e-6)
        rate = self.current / elapsed  # units/s
        remaining = max(self.total - self.current, 0)
        eta_s = remaining / rate if rate > 0 else float("inf")

        def _fmt(seconds: float) -> str:
            if seconds == float("inf"):
                return "?"
            seconds = int(seconds)
            return (f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
                    if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s")

        step_txt = f" step {self.step}" if self.step is not None else ""
        get_logger().info(
            f"[progress]{step_txt} {self.desc}: {self.current}/{self.total} "
            f"({percent}%) | {rate * 60:.1f}/min | elapsed {_fmt(elapsed)} | "
            f"eta {_fmt(eta_s)}")
        self._last_logged_percent = percent
        self._last_line_time = now

    def stall_tick(self, inflight: int = 0, scoring: int = 0):
        """Heartbeat when NO work completed in the wait window (2026-07-29).

        update() only logs on completions, so a stalled tail was silent
        exactly when visibility mattered most. Rate-limited to one line per
        heartbeat window.
        """
        now = self._time.time() if hasattr(self, "_time") else None
        if now is None:
            return
        if now - self._last_line_time < self._heartbeat_s:
            return
        elapsed = max(now - self._start_time, 1e-6)
        step_txt = f" step {self.step}" if self.step is not None else ""
        get_logger().info(
            f"[progress]{step_txt} {self.desc}: {self.current}/{self.total} "
            f"— NO completions in the last {int(self._heartbeat_s)}s "
            f"(inflight {inflight}, scoring {scoring}, "
            f"elapsed {int(elapsed) // 60}m{int(elapsed) % 60:02d}s)")
        self._last_line_time = now

    def set_postfix(self, postfix: dict[str, Any]):
        self._postfix = postfix
        if self._pbar is not None:
            self._pbar.set_postfix(postfix)

    def _log_progress(self):
        percent = int(100 * self.current / self.total) if self.total > 0 else 0
        if percent >= self._last_logged_percent + self.log_every_percent or self.current >= self.total:
            self._emit_progress(percent)
            self._last_logged_percent = percent

    def _emit_progress(self, percent: int):
        """Emit progress as structured JSON through loguru (only called in JSON logging mode)."""
        get_logger().bind(
            _progress=True,
            desc=self.desc,
            current=self.current,
            total=self.total,
            percent=percent,
            step=self.step,
            postfix=self._postfix if self._postfix else None,
        ).info("progress")

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
        elif self.current > 0 and self.current < self.total:
            percent = int(100 * self.current / self.total)
            if percent > self._last_logged_percent:
                self._emit_progress(percent)
