# stdlib
import logging
from pathlib import Path
from typing import Any

# third-party
from rich.logging import RichHandler

LOGGER_NAME = "myagent"
UNSET = object()

LOG_LEVEL_MAP: dict[str, int] = {
    "debug":    logging.DEBUG,
    "info":     logging.INFO,
    "warning":  logging.WARNING,
    "error":    logging.ERROR,
    "critical": logging.CRITICAL,
}

def setup_logger(console_log:bool = False, log_level: str = "debug" ) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LOG_LEVEL_MAP[log_level])
    logger.propagate = False

    if logger.handlers:
        return logger
    #if we need the console log we add the handler..
    if console_log:
        handler = RichHandler(
            show_path=False,
            show_time=False,
            show_level=False,
            markup=True,
        )
        formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def add_file_handler(path: Path | str, level: int = logging.DEBUG) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)-30s - %(levelname)-5s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def recursive_merge(*dictionaries: dict | None) -> dict:
    """Merge multiple dictionaries recursively.

    Later dictionaries take precedence over earlier ones.
    Nested dictionaries are merged recursively.
    UNSET values are skipped.
    """
    if not dictionaries:
        return {}
    result: dict[str, Any] = {}
    for d in dictionaries:
        if d is None:
            continue
        for key, value in d.items():
            if value is UNSET:
                continue
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge(result[key], value)
            elif isinstance(value, dict):
                # Recursively merge dict values to filter out nested UNSET values
                result[key] = recursive_merge(value)
            else:
                result[key] = value
    return result