"""Structured logging module for m.AI.leage."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format logs as compact, single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry["data"] = record.extra_data
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logger(
    name: str = "mileage",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    structured_json: bool = False,
) -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console Handler (minimal or JSON)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    if structured_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        plain_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(plain_formatter)

    # By default, do not spam standard stdout with general INFO logs;
    # Keep console handler at WARNING or above unless DEBUG is set
    if numeric_level > logging.DEBUG:
        console_handler.setLevel(logging.WARNING)

    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JSONFormatter() if structured_json else logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(file_handler)
        except Exception:
            # Silently degrade if file logging cannot be initialized
            pass

    return logger


# Default logger instance
logger = setup_logger("mileage")
