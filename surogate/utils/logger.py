import logging
import inspect
from typing import Optional
from swift.utils import get_logger as swift_get_logger


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
    """Wrapper around swift logger with colors and file/line info."""

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


def get_logger(name: Optional[str] = None, show_location: bool = True):
    """
    Get logger with colors and file/line number support.

    Args:
        name: Logger name (auto-detected if None)
        show_location: Whether to show file and line number

    Returns:
        LoggerWrapper instance
    """
    if name is None:
        # Auto-detect calling module
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'surogate')

    logger = swift_get_logger(name)
    return LoggerWrapper(logger, show_location=show_location)