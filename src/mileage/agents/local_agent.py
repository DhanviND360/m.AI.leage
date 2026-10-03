"""Local execution agent integrating Ollama, Workspace context, and Metrics."""

import time
from typing import Callable, Iterator, Optional

from mileage.agents.base import BaseAgent
from mileage.core.logger import logger
from mileage.core.workspace import WorkspaceManager
from mileage.metrics.schemas import MetricRecord
from mileage.metrics.tracker import MetricsTracker
from mileage.models.ollama_client import OllamaClient
from mileage.models.schemas import ChatMessage, MessageRole


class LocalAgent(BaseAgent):
    """Local-first agent executing tasks with local Ollama models."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model_name: str,
        workspace: Optional[WorkspaceManager] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name="LocalAgent", system_prompt=system_prompt)
        self.client = ollama_client
        self.model_name = model_name
        self.workspace = workspace
        self.metrics = metrics_tracker
        self.temperature = temperature

    def inject_workspace_context(self) -> None:
        """Inject summary of workspace files into conversation context."""
        if not self.workspace:
            return

        try:
            stats = self.workspace.get_stats()
            context_text = (
                f"[Workspace Context]\n"
                f"Project: {stats.project_name}\n"
                f"Total files: {stats.total_files}\n"
                f"Total bytes: {stats.total_bytes:,}\n"
                f"Extension breakdown: {dict(stats.extension_counts)}\n"
            )
            self.messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=context_text)
            )
        except Exception as e:
            logger.debug("Could not inject workspace context: %s", e)

    def run(self, prompt: str) -> str:
        """Execute a prompt synchronously, returning the response text."""
        self.add_user_message(prompt)
        start_time = time.perf_counter()
        success = True
        error_type = None
        response_text = ""
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = 0

        try:
            raw_resp = self.client.chat(
                messages=self.messages,
                model=self.model_name,
                temperature=self.temperature,
            )
            msg = raw_resp.get("message", {})
            response_text = msg.get("content", "")
            self.add_assistant_message(response_text)

            prompt_tokens = raw_resp.get("prompt_eval_count", prompt_tokens)
            completion_tokens = raw_resp.get("eval_count", max(1, len(response_text) // 4))

        except Exception as e:
            success = False
            error_type = type(e).__name__
            logger.error("Agent execution failed: %s", e)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            if self.metrics:
                self.metrics.record(
                    MetricRecord(
                        command="run",
                        model=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        latency_ms=round(latency_ms, 2),
                        success=success,
                        error_type=error_type,
                    )
                )

        return response_text

    def stream_run(
        self, prompt: str, on_first_token: Optional[Callable[[], None]] = None
    ) -> Iterator[str]:
        """Stream response chunks while capturing timing and token metrics."""
        self.add_user_message(prompt)
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        full_response: list[str] = []
        success = True
        error_type = None
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = 0

        try:
            for chunk in self.client.stream_chat(
                messages=self.messages,
                model=self.model_name,
                temperature=self.temperature,
            ):
                if first_token_time is None:
                    first_token_time = (time.perf_counter() - start_time) * 1000
                    if on_first_token:
                        on_first_token()

                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    full_response.append(delta)
                    yield delta

                # Check if Ollama provided token metrics in the final chunk
                if chunk.get("done", False):
                    prompt_tokens = chunk.get("prompt_eval_count", prompt_tokens)
                    completion_tokens = chunk.get("eval_count", completion_tokens)

            complete_str = "".join(full_response)
            self.add_assistant_message(complete_str)
            if not completion_tokens:
                completion_tokens = max(1, len(complete_str) // 4)

        except Exception as e:
            success = False
            error_type = type(e).__name__
            logger.error("Agent stream failed: %s", e)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            if self.metrics:
                self.metrics.record(
                    MetricRecord(
                        command="stream_run",
                        model=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        latency_ms=round(latency_ms, 2),
                        time_to_first_token_ms=round(first_token_time, 2)
                        if first_token_time
                        else None,
                        success=success,
                        error_type=error_type,
                    )
                )
