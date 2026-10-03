"""Pydantic schemas for the Gemma-powered planner's structured output.

Every planner invocation converts raw input (text, voice transcript,
images, file contents) into a strict ActionPlan schema. The planner
NEVER directly modifies project files — it only emits structured JSON
plans for downstream agents to execute.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class Complexity(str, Enum):
    """Estimated plan complexity — drives downstream resource allocation."""

    TRIVIAL = "trivial"       # < 5 min, single-file change
    LOW = "low"               # 5–30 min, few files
    MEDIUM = "medium"         # 30–120 min, multi-file
    HIGH = "high"             # > 2 hrs, architectural
    CRITICAL = "critical"     # refactor / breaking change


class InputType(str, Enum):
    """Type of input provided to the planner."""

    TEXT = "text"
    VOICE_TRANSCRIPT = "voice_transcript"
    IMAGE = "image"
    FILE = "file"
    MIXED = "mixed"


class PlannerFileRef(BaseModel):
    """A file reference extracted by the planner — never written to directly."""

    path: str = Field(..., description="Relative workspace path to the file.")
    role: str = Field(
        "context",
        description="Role of this file: 'target' (to be modified), 'context' (for reference), 'output' (new file).",
    )
    summary: Optional[str] = Field(
        None, description="Brief summary of file contents or purpose."
    )
    estimated_tokens: Optional[int] = Field(
        None, description="Estimated token count for this file."
    )


class Requirement(BaseModel):
    """A single requirement extracted from the user's input."""

    id: str = Field(..., description="Short identifier, e.g. R1, R2.")
    description: str = Field(..., description="Clear, actionable requirement.")
    priority: str = Field(
        "must", description="Priority: 'must', 'should', 'could'."
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        allowed = {"must", "should", "could"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            return "must"
        return v_lower


class Constraint(BaseModel):
    """A constraint or limitation the plan must respect."""

    description: str = Field(..., description="What must NOT happen or what is limited.")
    reason: Optional[str] = Field(
        None, description="Why this constraint exists."
    )


class AcceptanceCriterion(BaseModel):
    """A measurable condition for plan completion."""

    id: str = Field(..., description="Short identifier, e.g. AC1, AC2.")
    description: str = Field(..., description="Testable acceptance condition.")
    verification: Optional[str] = Field(
        None, description="How to verify this criterion was met."
    )


class ActionPlan(BaseModel):
    """The primary structured output of the Gemma planner.

    Represents a complete, validated action plan derived from user input.
    Downstream agents consume this schema to execute changes — the planner
    itself never touches project files.
    """

    model_config = {"protected_namespaces": ()}

    # Metadata
    plan_id: str = Field(
        default_factory=lambda: f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        description="Unique plan identifier.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 creation timestamp.",
    )
    input_type: InputType = Field(
        InputType.TEXT, description="Type of input that generated this plan."
    )

    # Core plan fields
    goal: str = Field(
        ..., description="One-sentence summary of what needs to be accomplished."
    )
    context: Optional[str] = Field(
        None, description="Brief background or context for the goal."
    )
    requirements: List[Requirement] = Field(
        default_factory=list, description="Extracted requirements."
    )
    files: List[PlannerFileRef] = Field(
        default_factory=list,
        description="Files relevant to this plan (targets, context, outputs).",
    )
    constraints: List[Constraint] = Field(
        default_factory=list, description="Constraints the implementation must respect."
    )
    complexity: Complexity = Field(
        Complexity.MEDIUM, description="Estimated complexity."
    )
    acceptance_criteria: List[AcceptanceCriterion] = Field(
        default_factory=list, description="Conditions that define 'done'."
    )

    # Planner metadata
    model_used: str = Field(
        "", description="Name of the Ollama model that generated this plan."
    )
    raw_input: Optional[str] = Field(
        None, description="Original raw input text (for traceability)."
    )
    image_analysis: Optional[str] = Field(
        None, description="Description from image analysis, if an image was provided."
    )
    tokens_used: int = Field(
        0, description="Total tokens consumed generating this plan."
    )
    latency_ms: float = Field(
        0.0, description="Wall-clock time for plan generation in milliseconds."
    )


class PlannerInput(BaseModel):
    """Structured input to the planner — supports text, images, and files."""

    text: Optional[str] = Field(
        None, description="Text prompt or voice transcript."
    )
    image_paths: List[str] = Field(
        default_factory=list, description="Absolute paths to images for multimodal analysis."
    )
    file_paths: List[str] = Field(
        default_factory=list, description="Workspace-relative paths to inject as context."
    )
    workspace_context: Optional[str] = Field(
        None, description="Pre-built workspace summary string."
    )
    max_context_tokens: int = Field(
        4096, description="Maximum token budget for file context injection."
    )

    @property
    def input_type(self) -> InputType:
        """Determine the dominant input type."""
        has_text = bool(self.text)
        has_images = bool(self.image_paths)
        has_files = bool(self.file_paths)

        if has_text and has_images:
            return InputType.MIXED
        if has_images:
            return InputType.IMAGE
        if has_files and has_text:
            return InputType.MIXED
        if has_files:
            return InputType.FILE
        return InputType.TEXT
