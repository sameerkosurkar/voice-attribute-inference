"""Structured logging.

PII rule, enforced here rather than left to reviewer discipline: the only audio
facts that may reach a log line are *scalars derived from* the audio -- byte
count, duration, SNR, quality verdict, timings. Never bytes, never a filename,
never a transcript. `scrub_processor` is a belt-and-braces guard that drops any
event key that looks like it carries a payload.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_FORBIDDEN_KEYS = {
    "audio", "audio_bytes", "pcm", "samples", "waveform",
    "payload", "body", "raw", "transcript", "text", "filename", "file_name",
}


def scrub_processor(_logger: Any, _name: str, event_dict: dict) -> dict:
    for key in list(event_dict):
        if key in _FORBIDDEN_KEYS:
            event_dict[key] = "<redacted>"
        elif isinstance(event_dict[key], (bytes, bytearray, memoryview)):
            event_dict[key] = f"<{len(event_dict[key])} bytes redacted>"
    return event_dict


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )
    # uvicorn's access log would record the request line for every /analyze
    # call. Harmless today, but it is the kind of thing that later grows a
    # query string with a phone number in it. We emit our own access event.
    logging.getLogger("uvicorn.access").disabled = True

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            scrub_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
