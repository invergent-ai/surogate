import os
from contextlib import contextmanager
from types import MethodType
import importlib
import logging
import inspect
from typing import Optional

init_loggers = {}
logger_format = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')
info_set = set()
warning_set = set()

# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

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
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


class LoggerWrapper:
    """Wrapper around logger with colors and file/line info."""

    def __init__(self, logger, show_location: bool = True):
        self._logger = logger
        self.show_location = show_location

    def _add_location(self, msg: str) -> str:
        """Add file and line number to message."""
        if not self.show_location:
            return msg

        frame = inspect.currentframe().f_back.f_back
        filename = frame.f_code.co_filename.split('/')[-1]
        lineno = frame.f_lineno
        location = f"{Colors.DIM}[{filename}:{lineno}]{Colors.RESET}"
        return f"{location} {msg}"

    def info(self, msg: str, *args, exc_info=None, **kwargs):
        """Log info message in cyan."""
        colored_msg = f"{Colors.CYAN}{self._add_location(msg)}{Colors.RESET}"
        self._logger.info(colored_msg, *args, exc_info=exc_info, **kwargs)

    def debug(self, msg: str, *args, exc_info=None, **kwargs):
        """Log debug message in bright black (gray)."""
        colored_msg = f"{Colors.BRIGHT_BLACK}{self._add_location(msg)}{Colors.RESET}"
        self._logger.debug(colored_msg, *args, exc_info=exc_info, **kwargs)

    def warning(self, msg: str, *args, exc_info=None, **kwargs):
        """Log warning message in yellow."""
        colored_msg = f"{Colors.YELLOW}{self._add_location(msg)}{Colors.RESET}"
        self._logger.warning(colored_msg, *args, exc_info=exc_info, **kwargs)

    def error(self, msg: str, *args, exc_info=None, **kwargs):
        """Log error message in red."""
        colored_msg = f"{Colors.RED}{self._add_location(msg)}{Colors.RESET}"
        self._logger.error(colored_msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args, exc_info=None, **kwargs):
        """Log critical message in bright red with bold."""
        colored_msg = f"{Colors.BOLD}{Colors.BRIGHT_RED}{self._add_location(msg)}{Colors.RESET}"
        self._logger.critical(colored_msg, *args, exc_info=exc_info, **kwargs)

    def success(self, msg: str, *args, exc_info=None, **kwargs):
        """Log success message in green."""
        colored_msg = f"{Colors.GREEN}✓ {self._add_location(msg)}{Colors.RESET}"
        self._logger.info(colored_msg, *args, exc_info=exc_info, **kwargs)

    def header(self, msg: str, *args, exc_info=None, **kwargs):
        """Log header message in bright blue with bold."""
        colored_msg = f"{Colors.BOLD}{Colors.BRIGHT_BLUE}{msg}{Colors.RESET}"
        self._logger.info(colored_msg, *args, exc_info=exc_info, **kwargs)

    def separator(self, char: str = "=", length: int = 60):
        """Log a separator line."""
        line = char * length
        colored_line = f"{Colors.BRIGHT_BLACK}{line}{Colors.RESET}"
        self._logger.info(colored_line)


def get_logger(log_file: Optional[str] = None, log_level: Optional[int] = None, file_mode: str = 'w'):
    """ Get logging logger

    Args:
        log_file: Log filename, if specified, file handler will be added to
            logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file, if filename is
            specified (if filemode is unspecified, it defaults to 'w').
    """
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level, logging.INFO)

    logger_name = __name__.split('.')[0]
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    if logger_name in init_loggers:
        add_file_handler_if_needed(logger, log_file, file_mode, log_level)
        return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    is_worker0 = _is_local_master()

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(logger_format)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_worker0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    init_loggers[logger_name] = True

    logger.info_once = MethodType(info_once, logger)
    logger.warning_once = MethodType(warning_once, logger)
    logger.info_if = MethodType(info_if, logger)
    logger.warning_if = MethodType(warning_if, logger)

    return LoggerWrapper(logger)

def info_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.info(msg)


def warning_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.warning(msg)


def info_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in info_set:
        return
    info_set.add(hash_id)
    self.info(msg)


def warning_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in warning_set:
        return
    warning_set.add(hash_id)
    self.warning(msg)


@contextmanager
def logger_context(logger, log_leval):
    origin_log_level = logger.level
    logger.setLevel(log_leval)
    try:
        yield
    finally:
        logger.setLevel(origin_log_level)

def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if importlib.util.find_spec('torch') is not None:
        is_worker0 = int(os.getenv('LOCAL_RANK', -1)) in {-1, 0}
    else:
        is_worker0 = True

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(logger_format)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

def _is_local_master():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    return local_rank in {-1, 0}

logger = get_logger()
logger._logger.handlers[0].setFormatter(logger_format)
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()