"""Configuration management using Pydantic."""

from pathlib import Path
from typing import List, Optional
import json
from pydantic import BaseModel, Field


class OllamaSettings(BaseModel):
    """Configuration for local Ollama instance."""

    host: str = Field(
        default="http://127.0.0.1:11434",
        description="Local Ollama server API base URL",
    )
    default_model: str = Field(
        default="llama3.2:latest",
        description="Default local model to use for agent execution",
    )
    fallback_model: Optional[str] = Field(
        default="llama3:latest",
        description="Fallback model if default is not installed",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds for Ollama requests",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for model generations",
    )


class WorkspaceSettings(BaseModel):
    """Configuration for local workspace management."""

    project_name: str = Field(
        default="mileage-project",
        description="Name of current initialized workspace",
    )
    mileage_dir: str = Field(
        default=".mileage",
        description="Local metadata and cache directory",
    )
    ignore_patterns: List[str] = Field(
        default_factory=lambda: [
            ".git",
            ".mileage",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "*.pyc",
            ".DS_Store",
            "dist",
            "build",
        ],
        description="Patterns to ignore when scanning workspace files",
    )
    max_scan_depth: int = Field(
        default=5,
        description="Maximum directory recursion depth for workspace scanning",
    )
    max_file_size_bytes: int = Field(
        default=1_000_000,  # 1MB
        description="Maximum file size in bytes to include in context scanning",
    )


class LoggingSettings(BaseModel):
    """Configuration for structured logging."""

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    structured_json: bool = Field(
        default=False,
        description="Whether to emit structured JSON logs to stderr/file",
    )
    log_to_file: bool = Field(
        default=True,
        description="Whether to write runtime logs into .mileage/logs/mileage.log",
    )


class MetricsSettings(BaseModel):
    """Configuration for local metrics and telemetry."""

    enabled: bool = Field(
        default=True,
        description="Enable local metrics logging",
    )
    metrics_filename: str = Field(
        default="metrics.jsonl",
        description="Filename inside .mileage to store JSONL execution metrics",
    )


class MileageConfig(BaseModel):
    """Top-level configuration schema for m.AI.leage."""

    version: str = Field(default="0.1.0", description="Configuration version")
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)

    @classmethod
    def load_from_dir(cls, directory: Path) -> "MileageConfig":
        """Load configuration from .mileage/config.json in the specified directory."""
        config_file = directory / ".mileage" / "config.json"
        if not config_file.exists():
            return cls()
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.model_validate(data)
        except Exception:
            # Fall back to default config if parse fails
            return cls()

    def save_to_dir(self, directory: Path) -> Path:
        """Save configuration to .mileage/config.json in the specified directory."""
        mileage_dir = directory / ".mileage"
        mileage_dir.mkdir(parents=True, exist_ok=True)
        config_file = mileage_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return config_file
