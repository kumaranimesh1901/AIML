"""Unit tests for structured logging."""

from __future__ import annotations

import json
import logging

from edg.config import LoggingConfig
from edg.logging import JsonFormatter, TextFormatter, get_logger, setup_logging


def test_json_formatter_outputs_valid_json() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="edg.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="step_done",
        args=(),
        exc_info=None,
    )
    record.run_id = "run_123"
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "step_done"
    assert parsed["run_id"] == "run_123"


def test_text_formatter_and_setup_logging() -> None:
    setup_logging(LoggingConfig(level="DEBUG", format="text"))
    logger = get_logger("edg.test")
    assert logger.getEffectiveLevel() == logging.DEBUG

    formatter = TextFormatter()
    record = logging.LogRecord(
        name="edg.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=20,
        msg="warn_msg",
        args=(),
        exc_info=None,
    )
    record.count = 5
    formatted = formatter.format(record)
    assert "[WARNING]" in formatted
    assert "warn_msg" in formatted
    assert '"count": 5' in formatted
