"""Custom typed exceptions for m.AI.leage."""

from typing import Optional


class MileageError(Exception):
    """Base exception for all m.AI.leage errors."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n💡 Hint: {self.suggestion}"
        return self.message


class WorkspaceError(MileageError):
    """Raised when workspace operations fail."""


class WorkspaceNotInitializedError(WorkspaceError):
    """Raised when an operation requires an initialized .mileage workspace."""

    def __init__(
        self,
        message: str = "Workspace is not initialized.",
        suggestion: str = "Run 'mileage init' to initialize m.AI.leage in this directory.",
    ):
        super().__init__(message=message, suggestion=suggestion)


class WorkspaceAlreadyInitializedError(WorkspaceError):
    """Raised when attempting to re-initialize an existing workspace without force."""

    def __init__(
        self,
        message: str = "Workspace is already initialized in this directory.",
        suggestion: str = "Use 'mileage init --force' if you wish to reinitialize.",
    ):
        super().__init__(message=message, suggestion=suggestion)


class OllamaError(MileageError):
    """Raised when Ollama interactions fail."""


class OllamaConnectionError(OllamaError):
    """Raised when unable to reach the local Ollama instance."""

    def __init__(
        self,
        message: str = "Unable to connect to local Ollama service.",
        suggestion: str = "Ensure Ollama is running locally. Start it with 'ollama serve' or launch the Ollama desktop app.",
    ):
        super().__init__(message=message, suggestion=suggestion)


class ModelNotFoundError(OllamaError):
    """Raised when the specified model is not installed locally in Ollama."""

    def __init__(
        self,
        model_name: str,
        suggestion: Optional[str] = None,
    ):
        msg = f"Model '{model_name}' was not found in local Ollama storage."
        sugg = suggestion or f"Pull the model first by running 'ollama pull {model_name}'."
        super().__init__(message=msg, suggestion=sugg)
        self.model_name = model_name


class ToolExecutionError(MileageError):
    """Raised when an agent tool fails during execution."""


class ToolPermissionError(MileageError):
    """Raised when a tool execution violates configured security permissions."""


class StagnationError(MileageError):
    """Raised when an agent execution loop reaches stagnation or retry limits."""


class ConfigValidationError(MileageError):
    """Raised when workspace or application configuration is invalid."""
