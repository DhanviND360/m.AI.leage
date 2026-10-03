"""Base interface for local-first agent tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result of executing an agent tool."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    output_str: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all m.AI.leage tools."""

    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool with provided arguments."""
        pass

    def to_ollama_definition(self) -> Dict[str, Any]:
        """Convert tool to format compatible with Ollama/LLM tool calling."""
        defn: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            },
        }
        if self.parameters:
            defn["function"]["parameters"] = self.parameters
        return defn
