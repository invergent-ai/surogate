import os
from contextlib import contextmanager
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
import logging
import inspect
from typing import Optional, Any

logging.captureWarnings(True)

init_loggers = {}
info_set = set()
warning_set = set()
DEFAULT_LOG_DIR = Path("logs")

# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    # Bright background colors
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'


class ColoredFormatter(logging.Formatter):
    """Formatter that preserves ANSI color codes and adds timestamps."""

    def __init__(self, fmt='%(message)s', datefmt='%H:%M:%S'):
        super().__init__(fmt, datefmt)

    def formatMessage(self, record):
        return record.getMessage()

    def format(self, record):
        message = self.formatMessage(record)

        # Add timestamp if configured
        if hasattr(self, '_show_timestamp') and self._show_timestamp:
            timestamp = self.formatTime(record, self.datefmt)
            colored_timestamp = f"{Colors.DIM}{timestamp}{Colors.RESET}"

            # Handle multi-line messages
            if '\n' in message:
                lines = message.split('\n')
                # Only add timestamp to first line
                return f"{colored_timestamp} {lines[0]}\n" + '\n'.join(lines[1:])
            else:
                return f"{colored_timestamp} {message}"

        return message


class LoggerWrapper:
    """Wrapper around logger with vibrant colors and file/line info."""

    def __init__(self, logger, show_location: bool = True):
        self._logger = logger
        self.show_location = show_location

    def _add_location(self, color: str) -> str:
        """Add file and line number with color."""
        if not self.show_location:
            return ""

        frame = inspect.currentframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        return f"{color}{Colors.ITALIC}[{filename}:{lineno}]{Colors.RESET} "


    def getEffectiveLevel(self):
        return self._logger.getEffectiveLevel()

    @property
    def level(self) -> int:
        return self._logger.level

    def info(self, msg: str, *args, exc_info=None, **kwargs):
        if _is_from_libraries():
            self.debug(msg, exc_info=exc_info, *args, **kwargs)
            return
        """Log info message in bright cyan."""
        prefix = f"{Colors.BRIGHT_CYAN}[INFO]{Colors.RESET}"
        colored_msg = f"{prefix} {self._add_location(Colors.BRIGHT_CYAN)}{Colors.WHITE}{msg}{Colors.RESET}"
        self._logger.info(colored_msg, *args, exc_info=exc_info, **kwargs)


    def info_once(self, msg: str, **kwargs):
        """Log info message only once."""
        hash_id = kwargs.get('hash_id') or msg
        if hash_id in info_set:
            return
        info_set.add(hash_id)
        self.info(msg, **kwargs)

    def info_if(self, msg: str, cond: bool):
        """Log info message if condition is true."""
        if cond:
            self.info(msg)

    def debug_once(self, msg: str, **kwargs):
        """Log info message only once."""
        hash_id = kwargs.get('hash_id') or msg
        if hash_id in info_set:
            return
        info_set.add(hash_id)
        self.debug(msg, **kwargs)

    def debug(self, msg: str, *args, exc_info=None, **kwargs):
        """Log debug message in magenta."""
        prefix = f"{Colors.MAGENTA}[DEBUG]{Colors.RESET}"
        colored_msg = f"{prefix} {self._add_location(Colors.MAGENTA)}{Colors.WHITE}{msg}{Colors.RESET}"
        self._logger.debug(colored_msg, *args, exc_info=exc_info, **kwargs)

    def warning(self, msg: str, *args, exc_info=None, **kwargs):
        if _is_from_libraries():
            self.debug(msg, exc_info=exc_info, *args, **kwargs)
            return
        """Log warning message in bright yellow."""
        prefix = f"{Colors.BRIGHT_YELLOW}[WARNING]{Colors.RESET}"
        colored_msg = f"{prefix} {self._add_location(Colors.BRIGHT_YELLOW)}{Colors.YELLOW}{msg}{Colors.RESET}"
        self._logger.warning(colored_msg, *args, exc_info=exc_info, **kwargs)

    def warning_once(self, msg: str, **kwargs):
        """Log warning message only once."""
        hash_id = kwargs.get('hash_id') or msg
        if hash_id in warning_set:
            return
        warning_set.add(hash_id)
        self.warning(msg, **kwargs)

    def warning_if(self, msg: str, cond: bool):
        """Log warning message if condition is true."""
        if cond:
            self.warning(msg)

    def error(self, msg: str, *args, exc_info=None, **kwargs):
        if _is_from_libraries():
            self.debug(msg, exc_info=exc_info, *args, **kwargs)
            return
        """Log error message in bright red."""
        prefix = f"{Colors.BRIGHT_RED}[ERROR]{Colors.RESET}"
        colored_msg = f"{prefix} {self._add_location(Colors.BRIGHT_RED)}{Colors.RED}{msg}{Colors.RESET}"
        self._logger.error(colored_msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args, exc_info=None, **kwargs):
        """Log critical message in bold bright red."""
        prefix = f"{Colors.BOLD}{Colors.BRIGHT_RED}[CRITICAL]{Colors.RESET}"
        colored_msg = f"{prefix} {self._add_location(Colors.BRIGHT_RED)}{Colors.RED}{msg}{Colors.RESET}"
        self._logger.critical(colored_msg, *args, exc_info=exc_info, **kwargs)

    def success(self, msg: str, *args, exc_info=None, **kwargs):
        if _is_from_libraries():
            self.debug(msg, exc_info=exc_info, *args, **kwargs)
            return
        """Log success message in bright green with checkmark."""
        prefix = f"{Colors.BRIGHT_GREEN}[SUCCESS]{Colors.RESET}"
        colored_msg = f"{prefix} {self._add_location(Colors.BRIGHT_GREEN)}{Colors.WHITE}{msg}{Colors.RESET}"
        self._logger.info(colored_msg, *args, exc_info=exc_info, **kwargs)

    def header(self, msg: str, *args, exc_info=None, **kwargs):
        if _is_from_libraries():
            self.debug(msg, exc_info=exc_info, *args, **kwargs)
            return
        """Log header message in bold bright blue."""
        colored_msg = f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}{'─' * 3} {msg} {'─' * 3}{Colors.RESET}"
        self._logger.info(colored_msg, *args, exc_info=exc_info, **kwargs)

    def separator(self, char: str = "─", length: int = 80, color: str = None):
        """Log a separator line."""
        line = char * length
        color_code = color or Colors.BRIGHT_BLACK
        colored_line = f"{color_code}{line}{Colors.RESET}"
        self._logger.info(colored_line)

    def highlight(self, msg: str, *args, exc_info=None, **kwargs):
        """Log highlighted message in bright magenta with background."""
        colored_msg = f"{Colors.BG_BRIGHT_MAGENTA}{Colors.BRIGHT_WHITE}{Colors.BOLD} {msg} {Colors.RESET}"
        self._logger.info(colored_msg, *args, exc_info=exc_info, **kwargs)

    def metric(self, name: str, value: Any, unit: str = "", *args, **kwargs):
        """Log a metric with special formatting."""
        metric_msg = f"{Colors.BRIGHT_CYAN}📊 {name}:{Colors.RESET} {Colors.BRIGHT_GREEN}{Colors.BOLD}{value}{Colors.RESET}"
        if unit:
            metric_msg += f" {Colors.DIM}{unit}{Colors.RESET}"
        self._logger.info(metric_msg, *args, **kwargs)

    def metrics(self, names: list, values: list, units: list = None, *args, **kwargs):
        """Log multiple metrics on a single line with special formatting."""
        if units is None:
            units = [""] * len(names)

        if len(names) != len(values):
            raise ValueError(f"Length mismatch: {len(names)} names vs {len(values)} values")

        if len(units) != len(names):
            raise ValueError(f"Length mismatch: {len(names)} names vs {len(units)} units")

        metric_parts = []
        for name, value, unit in zip(names, values, units):
            metric_str = f"{Colors.BRIGHT_CYAN}{name}:{Colors.RESET} {Colors.BRIGHT_GREEN}{Colors.BOLD}{value}{Colors.RESET}"
            if unit:
                metric_str += f" {Colors.DIM}{unit}{Colors.RESET}"
            metric_parts.append(metric_str)

        separator = f" {Colors.DIM}|{Colors.RESET} "
        metrics_msg = f"{Colors.BRIGHT_CYAN}📊{Colors.RESET} " + separator.join(metric_parts)
        self._logger.info(metrics_msg, *args, **kwargs)


    def step(self, step_num: int, total: int, msg: str, *args, **kwargs):
        """Log a step in a process."""
        progress = f"{Colors.BRIGHT_BLUE}{Colors.BOLD}[{step_num}/{total}]{Colors.RESET}"
        colored_msg = f"{progress} {Colors.CYAN}▸{Colors.RESET} {msg}"
        self._logger.info(colored_msg, *args, **kwargs)

    def banner(self, msg: str, char: str = "═", *args, **kwargs):
        """Log a banner message."""
        length = max(len(msg) + 4, 60)
        top_border = f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{char * length}{Colors.RESET}"
        middle = f"{Colors.BRIGHT_BLUE}{Colors.BOLD}║{Colors.RESET} {Colors.BRIGHT_WHITE}{Colors.BOLD}{msg.center(length - 4)}{Colors.RESET} {Colors.BRIGHT_BLUE}{Colors.BOLD}║{Colors.RESET}"
        bottom_border = f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{char * length}{Colors.RESET}"

        # Log as single multi-line message
        full_banner = f"\n{top_border}\n{middle}\n{bottom_border}"
        self._logger.info(full_banner)

    def addFilter(self, filter):
        pass

def get_logger(
        name: Optional[str] = None,
        log_file: Optional[str] = None,
        log_level: Optional[int] = None,
        file_mode: str = 'w',
        enable_file_logging: bool = False,
        show_timestamp: bool = True  # NEW: control timestamps
):
    """Get logging logger with vibrant color support

    Args:
        log_file: Log filename. If None and enable_file_logging=True,
                  auto-generates filename in logs/ directory
        log_level: Logging level.
        file_mode: File open mode (default: 'w').
        enable_file_logging: Whether to enable file logging (default: False)
        show_timestamp: Whether to show timestamps (default: True)
    """

    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level, logging.INFO)

    logger_name = (name or __name__).split('.')[0]
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    if logger_name in init_loggers:
        add_file_handler_if_needed(logger, log_file, file_mode, log_level)
        return LoggerWrapper(logger)

    # Handle duplicate logs to the console
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    is_worker0 = _is_local_master()

    # File logging setup
    if is_worker0 and enable_file_logging:
        if log_file is None:
            # Auto-generate log filename
            DEFAULT_LOG_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(DEFAULT_LOG_DIR / f"surogate_{timestamp}.log")
        else:
            # Ensure parent directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = str(log_path)

        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    # Use ColoredFormatter with or without timestamp
    if show_timestamp:
        colored_formatter = ColoredFormatter(datefmt='%Y-%m-%d %H:%M:%S')  # ← CHANGED HERE
        colored_formatter._show_timestamp = True
    else:
        colored_formatter = ColoredFormatter()
        colored_formatter._show_timestamp = False

    for handler in handlers:
        handler.setFormatter(colored_formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_worker0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    init_loggers[logger_name] = True

    wrapper = LoggerWrapper(logger)
    return wrapper

@contextmanager
def logger_context(logger, log_level):
    origin_log_level = logger.level
    logger.setLevel(log_level)
    try:
        yield
    finally:
        logger.setLevel(origin_log_level)


def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if find_spec('torch') is not None:
        is_worker0 = int(os.getenv('LOCAL_RANK', -1)) in {-1, 0}
    else:
        is_worker0 = True

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(ColoredFormatter('%(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)


def _is_local_master():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    return local_rank in {-1, 0}

def _is_from_libraries():
    frame = inspect.currentframe()
    try:
        # Go up the stack to find the actual caller (skip wrapper frames)
        caller_frame = frame.f_back
        while caller_frame:
            caller_module = caller_frame.f_globals.get('__name__', '')
            # Check if we're still in the logger wrapper
            if not caller_module.startswith(__name__):
                break
            caller_frame = caller_frame.f_back

        if caller_module.startswith('transformers') or caller_module.startswith('datasets') or caller_module.startswith('huggingface_hub'):
            return True
    finally:
        del frame  # Avoid reference cycles

    return False

# Create module-level logger
logger = get_logger()

os.environ['HF_HUB_VERBOSITY'] = 'error'
import huggingface_hub
huggingface_hub.logging.get_logger = get_logger
huggingface_hub.utils.logging.get_logger = get_logger

from transformers.utils.logging import EmptyTqdm
import transformers.utils.logging
transformers.utils.logging.get_logger = get_logger

import datasets
datasets.utils.logging.get_logger = get_logger


# Test the logger if run directly
if __name__ == "__main__":
    # Clear any existing logger first
    logger_name = __name__.split('.')[0]
    if logger_name in init_loggers:
        del init_loggers[logger_name]
        logging.getLogger(logger_name).handlers.clear()

    test_logger = get_logger(log_level=logging.DEBUG)

    test_logger.banner("SUROGATE LOGGER TEST")

    test_logger.info("This is an info message with location")
    test_logger.debug("This is a debug message for troubleshooting")
    test_logger.success("Operation completed successfully!")
    test_logger.warning("This is a warning - pay attention!")
    test_logger.error("Something went wrong here")
    test_logger.critical("CRITICAL ERROR - immediate attention required!")

    test_logger.separator()

    test_logger.header("Metrics Section")
    test_logger.metric("Accuracy", "95.3%")
    test_logger.metric("Latency", "45", "ms")
    test_logger.metric("Throughput", "1000", "req/s")
    test_logger.metrics(
        names=["Accuracy", "Latency", "Throughput"],
        values=["95.3%", "45", "1000"],
        units=["", "ms", "req/s"]
    )

    test_logger.separator()

    test_logger.header("Step-by-Step Process")
    test_logger.step(1, 5, "Loading dataset...")
    test_logger.step(2, 5, "Preprocessing data...")
    test_logger.step(3, 5, "Training model...")
    test_logger.step(4, 5, "Evaluating results...")
    test_logger.step(5, 5, "Saving outputs...")

    test_logger.separator(char="═", color=Colors.BRIGHT_GREEN)

    test_logger.highlight("HIGHLIGHTED MESSAGE")

    test_logger.separator()