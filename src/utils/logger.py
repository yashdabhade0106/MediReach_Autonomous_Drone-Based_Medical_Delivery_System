# -*- coding: utf-8 -*-
"""
MediReach — Centralised Logging Configuration.

Provides a structured, rotating-file logger used by every module.
All modules should call ``get_logger(__name__)`` instead of
configuring their own ``logging.getLogger`` instances.

Features:
    - Rotating file handler (10 MB max, 5 backups)
    - Console handler with coloured output
    - JSON-structured log format for production
    - Configurable via environment variables
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════
#  Formatters
# ═══════════════════════════════════════════════════════════

class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-encoded log entry string.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data  # type: ignore[attr-defined]

        return json.dumps(log_entry, default=str)


class ColouredConsoleFormatter(logging.Formatter):
    """Coloured console formatter for human-readable dev output."""

    COLOURS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with ANSI colour codes.

        Args:
            record: The log record to format.

        Returns:
            Colour-coded log string.
        """
        colour = self.COLOURS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = (
            f"{colour}[{timestamp}] "
            f"{record.levelname:<8}{self.RESET} "
            f"\033[35m{record.name}\033[0m — "
            f"{record.getMessage()}"
        )
        if record.exc_info and record.exc_info[0] is not None:
            formatted += f"\n{self.formatException(record.exc_info)}"
        return formatted


# ═══════════════════════════════════════════════════════════
#  Logger Factory
# ═══════════════════════════════════════════════════════════

_LOGGERS_CONFIGURED: set = set()


def get_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    use_json: bool = False,
) -> logging.Logger:
    """Create or retrieve a configured logger instance.

    Ensures each named logger is configured exactly once
    to avoid duplicate handlers on repeated calls.

    Args:
        name: Logger name, typically ``__name__``.
        log_level: Override log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
            Falls back to ``LOG_LEVEL`` env var, then ``INFO``.
        log_file: Override log file path. Falls back to ``LOG_FILE``
            env var, then ``logs/medireach.log``.
        use_json: If True, file handler uses JSON formatter.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    if name in _LOGGERS_CONFIGURED:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    # Resolve log level
    level_str = log_level or os.getenv("LOG_LEVEL", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    # Prevent propagation to root to avoid duplicate output
    logger.propagate = False

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColouredConsoleFormatter())
    logger.addHandler(console_handler)

    # --- File handler ---
    file_path = log_file or os.getenv("LOG_FILE", "logs/medireach.log")
    max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10 MB
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    try:
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)

        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not create file handler at %s: %s", file_path, exc)

    _LOGGERS_CONFIGURED.add(name)
    return logger


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **extra: object,
) -> None:
    """Log a message with arbitrary key-value context.

    The extra data is attached to the log record and rendered
    by the JSON formatter in production.

    Args:
        logger: Logger instance.
        level: Logging level constant (e.g. ``logging.INFO``).
        message: Human-readable log message.
        **extra: Arbitrary context key-value pairs.
    """
    record = logger.makeRecord(
        name=logger.name,
        level=level,
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.extra_data = extra  # type: ignore[attr-defined]
    logger.handle(record)
