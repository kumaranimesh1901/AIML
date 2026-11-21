"""Structured logging setup for EDG."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from edg.config import LoggingConfig

_STANDARD_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON with structured extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable structured text formatter."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_RECORD_ATTRS and not k.startswith("_")
        }
        extra_str = f" | {json.dumps(extras, default=str)}" if extras else ""
        base = f"{ts} [{record.levelname}] {record.name}: {record.getMessage()}{extra_str}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure root logger for structured output."""
    cfg = config or LoggingConfig()
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.level, logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    formatter: logging.Formatter = JsonFormatter() if cfg.format == "json" else TextFormatter()
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
