import logging
import os
import sys
from typing import TextIO


_RESET = "\033[0m"
_DIM = "\033[2m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35m",
}


class HumanReadableFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname.lower().ljust(8)
        name = _short_logger_name(record.name)
        message = record.getMessage()

        if self._use_color:
            level_color = _LEVEL_COLORS.get(record.levelno, "")
            timestamp = f"{_DIM}{timestamp}{_RESET}"
            level = f"{level_color}{level}{_RESET}"
            name = f"{_DIM}{name}{_RESET}"

        formatted = f"{timestamp} {level} {name} {message}"
        if record.exc_info:
            formatted = f"{formatted}\n{self.formatException(record.exc_info)}"
        return formatted


def configure_logging(*, level: str | int | None = None, stream: TextIO | None = None) -> None:
    stream = stream or sys.stderr
    resolved_level = _resolve_level(level or os.environ.get("LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler(stream)
    handler.setFormatter(HumanReadableFormatter(use_color=_should_use_color(stream)))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(level.upper())
    return value if isinstance(value, int) else logging.INFO


def _should_use_color(stream: TextIO) -> bool:
    mode = os.environ.get("LOG_COLOR", "auto").lower()
    if mode in {"1", "true", "yes", "always"}:
        return True
    if mode in {"0", "false", "no", "never"} or "NO_COLOR" in os.environ:
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _short_logger_name(name: str) -> str:
    prefix = "lang_graph_state."
    return name.removeprefix(prefix)
