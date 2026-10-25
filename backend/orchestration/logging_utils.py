"""Logging utilities with automatic secret redaction for CodeGhost Orchestration."""

import logging
import re
from typing import Optional

from backend.orchestration.config import OrchestrationConfig


class SecretRedactionFilter(logging.Filter):
    """Logging filter that scrubs sensitive API keys from log records."""

    def __init__(self, config: Optional[OrchestrationConfig] = None):
        super().__init__()
        self.config = config

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._redact(v) if isinstance(v, str) else v for v in record.args)
        return True

    def _redact(self, text: str) -> str:
        if self.config:
            text = self.config.redact(text)
        return re.sub(r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED_API_KEY]", text)


def get_orchestration_logger(name: str = "codeghost.orchestration", config: Optional[OrchestrationConfig] = None) -> logging.Logger:
    """Retrieve or configure the orchestration logger with secret scrubbing."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, SecretRedactionFilter) for f in logger.filters):
        logger.addFilter(SecretRedactionFilter(config))
    return logger
