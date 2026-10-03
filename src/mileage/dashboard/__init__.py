"""In-process event bus for broadcasting live agent events to SSE dashboard clients.

Thread-safe event bus that allows agents (running in the main thread) to publish
events, while the async SSE server reads and broadcasts them to connected browsers.
"""

import asyncio
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    """Stages in the m.AI.leage agent pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    PLANNING = "planning"
    ROUTING = "routing"
    BUILDING = "building"
    EVALUATING = "evaluating"
    ESCALATING = "escalating"
    COMPLETE = "complete"
    FAILED = "failed"


class DashboardEvent(BaseModel):
    """A single event broadcast to dashboard SSE clients."""

    event_type: str  # pipeline_update, metric, build_start, build_end, etc.
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data: Dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        """Format as an SSE data frame."""
        payload = self.model_dump_json()
        return f"data: {payload}\n\n"


class PipelineStatus(BaseModel):
    """Current state of the agent pipeline (read by dashboard)."""

    model_config = {"protected_namespaces": ()}

    stage: PipelineStage = PipelineStage.IDLE
    goal: str = ""
    model_name: str = ""
    iteration: int = 0
    max_iterations: int = 0
    message: str = ""
    requirements_passed: int = 0
    requirements_total: int = 0
    started_at: Optional[str] = None
    elapsed_ms: float = 0.0


class BuildRecord(BaseModel):
    """Summary of a completed build for the history view."""

    model_config = {"protected_namespaces": ()}

    build_id: str = ""
    goal: str = ""
    model_name: str = ""
    status: str = "complete"
    started_at: str = ""
    completed_at: str = ""
    duration_ms: float = 0.0
    tokens_used: int = 0
    requirements_passed: int = 0
    requirements_total: int = 0
    escalated: bool = False


class DashboardStats(BaseModel):
    """Aggregated statistics for the dashboard "today" cards."""

    projects_built: int = 0
    total_tokens_used: int = 0
    tokens_saved_estimate: int = 0
    estimated_cost_avoided: float = 0.0
    avg_requirement_satisfaction: float = 0.0
    tasks_completed_locally: float = 0.0
    total_builds_today: int = 0
    models_used: List[str] = Field(default_factory=list)
    avg_build_time_ms: float = 0.0


class DashboardEventBus:
    """Thread-safe event bus for broadcasting agent events to async SSE clients.

    Agents push events from sync code (main thread).
    The async SSE server consumes events from async subscriber queues.
    """

    _instance: Optional["DashboardEventBus"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._subscribers: List[asyncio.Queue] = []
        self._sub_lock = threading.Lock()
        self._pipeline_status = PipelineStatus()
        self._build_history: Deque[BuildRecord] = deque(maxlen=50)
        self._stats = DashboardStats()
        self._event_log: Deque[DashboardEvent] = deque(maxlen=200)
        self._load_state_from_disk()

    @property
    def state_file(self) -> Path:
        return Path.cwd() / ".mileage" / "dashboard_state.json"

    @property
    def events_file(self) -> Path:
        return Path.cwd() / ".mileage" / "dashboard_events.jsonl"

    def _save_state_to_disk(self) -> None:
        """Persist state to disk for cross-process synchronization."""
        try:
            p = self.state_file
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "pipeline": self._pipeline_status.model_dump(),
                "stats": self._stats.model_dump(),
                "history": [b.model_dump() for b in self._build_history],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
            tmp.replace(p)
        except Exception:
            pass

    def _load_state_from_disk(self) -> None:
        """Load state from disk if written by another process."""
        try:
            p = self.state_file
            if p.is_file():
                raw = p.read_text(encoding="utf-8")
                data = json.loads(raw)
                if "pipeline" in data and isinstance(data["pipeline"], dict):
                    self._pipeline_status = PipelineStatus.model_validate(data["pipeline"])
                if "stats" in data and isinstance(data["stats"], dict):
                    self._stats = DashboardStats.model_validate(data["stats"])
                if "history" in data and isinstance(data["history"], list):
                    self._build_history.clear()
                    for item in reversed(data["history"][:50]):
                        self._build_history.appendleft(BuildRecord.model_validate(item))
        except Exception:
            pass

    @classmethod
    def get_instance(cls) -> "DashboardEventBus":
        """Get or create the singleton event bus."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def clear(self) -> None:
        """Clear all in-memory and disk records (for testing)."""
        self._build_history.clear()
        self._event_log.clear()
        self._pipeline_status = PipelineStatus()
        self._stats = DashboardStats()
        try:
            if self.state_file.is_file():
                self.state_file.unlink(missing_ok=True)
            if self.events_file.is_file():
                self.events_file.unlink(missing_ok=True)
        except Exception:
            pass

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton and disk state (for testing)."""
        if cls._instance is not None:
            cls._instance.clear()
        else:
            try:
                sf = Path.cwd() / ".mileage" / "dashboard_state.json"
                if sf.is_file():
                    sf.unlink(missing_ok=True)
                ef = Path.cwd() / ".mileage" / "dashboard_events.jsonl"
                if ef.is_file():
                    ef.unlink(missing_ok=True)
            except Exception:
                pass
        cls._instance = None

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscriber queue for an SSE client."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def publish(self, event: DashboardEvent) -> None:
        """Publish an event to all subscribers (thread-safe, sync)."""
        self._event_log.append(event)

        # Append to shared event log for cross-process SSE tailing
        try:
            ef = self.events_file
            ef.parent.mkdir(parents=True, exist_ok=True)
            with open(ef, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except Exception:
            pass

        self._save_state_to_disk()

        with self._sub_lock:
            dead_queues = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    dead_queues.append(q)
            for dq in dead_queues:
                try:
                    self._subscribers.remove(dq)
                except ValueError:
                    pass

    def publish_pipeline_update(
        self,
        stage: PipelineStage,
        message: str = "",
        **kwargs,
    ) -> None:
        """Convenience: update pipeline status and broadcast."""
        self._pipeline_status.stage = stage
        self._pipeline_status.message = message
        for k, v in kwargs.items():
            if hasattr(self._pipeline_status, k):
                setattr(self._pipeline_status, k, v)

        self.publish(DashboardEvent(
            event_type="pipeline_update",
            data=self._pipeline_status.model_dump(),
        ))

    def record_build(self, build: BuildRecord) -> None:
        """Record a completed build and update stats."""
        self._build_history.appendleft(build)
        self._recompute_stats()
        self._save_state_to_disk()
        self.publish(DashboardEvent(
            event_type="build_complete",
            data=build.model_dump(),
        ))

    def _recompute_stats(self) -> None:
        """Recompute aggregate stats from build history."""
        builds = list(self._build_history)
        if not builds:
            return

        self._stats.projects_built = len(builds)
        self._stats.total_tokens_used = sum(b.tokens_used for b in builds)
        # Estimate: local tokens cost ~$0 vs cloud at ~$0.002/1k tokens
        self._stats.tokens_saved_estimate = self._stats.total_tokens_used
        self._stats.estimated_cost_avoided = round(
            self._stats.total_tokens_used * 0.002 / 1000, 2
        )

        sat_values = [
            b.requirements_passed / max(b.requirements_total, 1) * 100
            for b in builds
        ]
        self._stats.avg_requirement_satisfaction = round(
            sum(sat_values) / len(sat_values), 1
        ) if sat_values else 0.0

        local_count = sum(1 for b in builds if not b.escalated)
        self._stats.tasks_completed_locally = round(
            local_count / len(builds) * 100, 1
        ) if builds else 0.0

        self._stats.total_builds_today = len(builds)
        self._stats.models_used = list(set(b.model_name for b in builds if b.model_name))
        durations = [b.duration_ms for b in builds if b.duration_ms > 0]
        self._stats.avg_build_time_ms = round(
            sum(durations) / len(durations), 1
        ) if durations else 0.0

    @property
    def pipeline_status(self) -> PipelineStatus:
        self._load_state_from_disk()
        return self._pipeline_status

    @property
    def stats(self) -> DashboardStats:
        self._load_state_from_disk()
        return self._stats

    @property
    def build_history(self) -> List[BuildRecord]:
        self._load_state_from_disk()
        return list(self._build_history)

    @property
    def subscriber_count(self) -> int:
        with self._sub_lock:
            return len(self._subscribers)
