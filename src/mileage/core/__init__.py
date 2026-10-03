"""Core subsystem for m.AI.leage."""

from mileage.core.config import MileageConfig, OllamaSettings, WorkspaceSettings
from mileage.core.exceptions import (
    MileageError,
    WorkspaceError,
    WorkspaceNotInitializedError,
    WorkspaceAlreadyInitializedError,
    OllamaError,
    OllamaConnectionError,
    ModelNotFoundError,
    ToolExecutionError,
    ConfigValidationError,
)
from mileage.core.workspace import WorkspaceManager, WorkspaceStats, FileMetadata
from mileage.core.logger import setup_logger, logger

__all__ = [
    "MileageConfig",
    "OllamaSettings",
    "WorkspaceSettings",
    "MileageError",
    "WorkspaceError",
    "WorkspaceNotInitializedError",
    "WorkspaceAlreadyInitializedError",
    "OllamaError",
    "OllamaConnectionError",
    "ModelNotFoundError",
    "ToolExecutionError",
    "ConfigValidationError",
    "WorkspaceManager",
    "WorkspaceStats",
    "FileMetadata",
    "setup_logger",
    "logger",
]
