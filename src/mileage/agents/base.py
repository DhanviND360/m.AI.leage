"""Base class and protocol for local agents."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from mileage.models.schemas import ChatMessage, MessageRole
from mileage.tools.base import BaseTool


class BaseAgent(ABC):
    """Abstract agent with conversation history and tool management."""

    def __init__(
        self,
        name: str = "LocalAgent",
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt or (
            "You are m.AI.leage, a fast, lightweight local-first AI assistant. "
            "You run completely offline on local models via Ollama. "
            "Help the developer inspect, understand, and build within their workspace."
        )
        self.messages: List[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
        ]
        self.tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """Register a local tool available to this agent."""
        self.tools[tool.name] = tool

    def add_user_message(self, content: str) -> None:
        """Add a user message to message history."""
        self.messages.append(ChatMessage(role=MessageRole.USER, content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant response to message history."""
        self.messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))

    def clear_history(self) -> None:
        """Reset conversation history while preserving system prompt."""
        self.messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
        ]

    @abstractmethod
    def run(self, prompt: str) -> str:
        """Execute a prompt and return the assistant response."""
        pass
