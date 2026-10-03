"""Pydantic schemas for the Coding Agent execution loop, steps, and telemetry."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mileage.agents.stagnation import StagnationReport


class AgentPhase(str, Enum):
    """Phases in the autonomous coding agent execution loop."""

    INSPECT = "inspect"
    PLAN = "plan"
    EDIT = "edit"
    TEST = "test"
    REPAIR = "repair"
    COMPLETE = "complete"


class AgentStatus(str, Enum):
    """Final outcome status of an agent coding session."""

    RUNNING = "running"
    COMPLETED = "completed"
    STAGNATED = "stagnated"
    FAILED = "failed"


class CodingStepRecord(BaseModel):
    """Detailed record of an individual tool call or agent step."""

    step_index: int
    iteration: int
    phase: AgentPhase
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_output_summary: str = ""
    success: bool = True
    error_message: Optional[str] = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CodingSessionRecord(BaseModel):
    """Complete persistent session telemetry record for a coding task."""

    model_config = {"protected_namespaces": ()}

    session_id: str
    goal: str
    model_name: str
    status: AgentStatus = AgentStatus.RUNNING
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    total_iterations: int = 0
    total_steps: int = 0
    total_duration_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    files_modified: List[str] = Field(default_factory=list)
    tests_passed: bool = False
    test_summary: str = ""
    stagnation_report: Optional[StagnationReport] = None
    steps: List[CodingStepRecord] = Field(default_factory=list)
    final_summary: str = ""


class AgentEvent(BaseModel):
    """Event emitted during execution for concise real-time progress display."""

    event_type: str
    iteration: int = 0
    phase: AgentPhase = AgentPhase.INSPECT
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
