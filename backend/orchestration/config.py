"""Orchestration Configuration and Safety Limits for CodeGhost.

Manages environment variable loading, model selection, secret redaction,
and operational budgets (token, cost, time, and retry limits).
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load .env file safely if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


@dataclass
class OrchestrationConfig:
    """Configuration settings and safety boundaries for autonomous execution."""
    openai_api_key: Optional[str] = field(default=None)
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None
    
    # Operational budgets
    max_requests: int = 50
    max_total_tokens: int = 250_000
    max_cost_usd: float = 5.00
    per_command_timeout_seconds: float = 180.0
    max_repair_attempts_per_task: int = 3
    max_review_retries: int = 3
    max_consecutive_failures: int = 3
    max_changed_files_per_task: int = 10
    max_patch_size_bytes: int = 100_000

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    state_file: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "reports" / "validation" / "orchestration_state.json")
    reports_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "reports" / "validation")

    def __post_init__(self):
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if os.environ.get("OPENAI_MODEL"):
            self.openai_model = os.environ.get("OPENAI_MODEL")

    @property
    def has_api_key(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def masked_api_key(self) -> str:
        """Return safely masked representation of the API key for diagnostics."""
        if not self.openai_api_key:
            return "[NOT CONFIGURED]"
        k = self.openai_api_key.strip()
        if len(k) <= 12:
            return "[CONFIGURED]"
        return f"{k[:7]}...{k[-4:]}"

    def redact(self, text: str) -> str:
        """Scrub known API keys and sensitive tokens from text."""
        if not text:
            return ""
        scrubbed = text
        if self.openai_api_key and len(self.openai_api_key) > 8:
            scrubbed = scrubbed.replace(self.openai_api_key, "[REDACTED_OPENAI_KEY]")
        # Regex scrub for any lingering standard OpenAI sk-... tokens
        scrubbed = re.sub(r"sk-[a-zA-Z0-9_-]{20,}", "[REDACTED_API_KEY]", scrubbed)
        return scrubbed
