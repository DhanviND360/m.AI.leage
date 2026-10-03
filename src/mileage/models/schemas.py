"""Pydantic schemas for models, chat messages, and health checks."""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """An individual message in a conversation."""

    role: MessageRole
    content: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ModelDetails(BaseModel):
    """Details about a specific model variant."""

    format: Optional[str] = None
    family: Optional[str] = None
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None


class ModelInfo(BaseModel):
    """Normalized metadata for a local Ollama model."""

    model_config = {"protected_namespaces": ()}

    name: str
    model: str
    size_bytes: int = 0
    size_human: str = "0 B"
    digest: Optional[str] = None
    modified_at: Optional[str] = None
    details: Optional[ModelDetails] = None

    @classmethod
    def format_size(cls, size_bytes: int) -> str:
        """Format bytes into readable MB or GB string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class OllamaHealthStatus(BaseModel):
    """Status of the local Ollama daemon connection."""

    is_running: bool
    host: str
    version: Optional[str] = None
    response_time_ms: float = 0.0
    error_message: Optional[str] = None
    models_count: int = 0


class DoctorCheckStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class DoctorCheckItem(BaseModel):
    """A single diagnostic check item for 'mileage doctor'."""

    category: str
    name: str
    status: DoctorCheckStatus
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None


class DoctorReport(BaseModel):
    """Comprehensive diagnostic report returned by mileage doctor."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    python_version: str
    os_info: str
    checks: List[DoctorCheckItem] = Field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return all(c.status != DoctorCheckStatus.ERROR for c in self.checks)

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if c.status == DoctorCheckStatus.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.status == DoctorCheckStatus.WARNING)

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.checks if c.status == DoctorCheckStatus.OK)
