"""Pydantic schemas for the Model Router subsystem.

Defines capability categories, model profiles, routing decisions,
benchmark records, and configurable routing rules.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


# ── Capability Taxonomy ──────────────────────────────────────────────────────


class Capability(str, Enum):
    """Task capabilities a model may possess."""

    CODING = "coding"           # Code generation, debugging, refactoring
    REASONING = "reasoning"     # Logic, math, multi-step analysis
    VISION = "vision"           # Image understanding (multimodal)
    SPEED = "speed"             # Fast inference, low latency
    PLANNING = "planning"       # Structured plan generation
    CHAT = "chat"               # General conversation
    SUMMARIZATION = "summarization"  # Document / code summarization
    INSTRUCTION = "instruction"      # Precise instruction following


class TaskCategory(str, Enum):
    """High-level task categories for routing decisions."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    PLANNING = "planning"
    CHAT = "chat"
    SUMMARIZATION = "summarization"
    IMAGE_ANALYSIS = "image_analysis"
    REASONING = "reasoning"
    GENERAL = "general"


class CapabilityScore(BaseModel):
    """A model's score for a specific capability (0.0 – 1.0)."""

    capability: Capability
    score: float = Field(0.5, ge=0.0, le=1.0)
    source: str = Field(
        "heuristic",
        description="How this score was determined: 'heuristic', 'benchmark', 'manual'.",
    )


# ── Model Profile ───────────────────────────────────────────────────────────


class HardwareInfo(BaseModel):
    """Runtime / hardware context for the local system."""

    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_mb: Optional[int] = None
    cpu_cores: Optional[int] = None
    ram_total_mb: Optional[int] = None
    os_platform: Optional[str] = None


class ModelProfile(BaseModel):
    """Complete profile of an installed Ollama model with capabilities."""

    model_config = {"protected_namespaces": ()}

    # Identity
    name: str = Field(..., description="Full Ollama model name (e.g. 'gemma3:4b').")
    family: str = Field("unknown", description="Model family (e.g. 'gemma', 'llama', 'qwen').")
    parameter_size: Optional[str] = Field(None, description="Parameter count string (e.g. '4B', '8B').")
    quantization: Optional[str] = Field(None, description="Quantization level (e.g. 'Q4_K_M').")

    # Size & context
    size_bytes: int = Field(0, description="On-disk model size in bytes.")
    size_human: str = Field("0 B", description="Human-readable size string.")
    context_length: int = Field(
        4096, description="Maximum context window in tokens."
    )

    # Capabilities (populated by registry)
    capabilities: List[CapabilityScore] = Field(
        default_factory=list, description="Scored capabilities for this model."
    )

    # Performance (populated by benchmarks)
    avg_latency_ms: Optional[float] = Field(None, description="Average response latency from benchmarks.")
    avg_tokens_per_sec: Optional[float] = Field(None, description="Average token generation speed.")
    benchmark_success_rate: Optional[float] = Field(None, description="Success rate from benchmark tasks (0-1).")
    benchmark_run_count: int = Field(0, description="Number of benchmark runs completed.")

    # Metadata
    discovered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def get_capability_score(self, cap: Capability) -> float:
        """Get score for a specific capability, defaulting to 0.0."""
        for cs in self.capabilities:
            if cs.capability == cap:
                return cs.score
        return 0.0

    def has_capability(self, cap: Capability, threshold: float = 0.3) -> bool:
        """Check if model meets the minimum threshold for a capability."""
        return self.get_capability_score(cap) >= threshold

    @property
    def parameter_count_billions(self) -> Optional[float]:
        """Parse parameter size string to numeric billions."""
        if not self.parameter_size:
            return None
        ps = self.parameter_size.upper().strip()
        try:
            if ps.endswith("B"):
                return float(ps[:-1])
            elif ps.endswith("M"):
                return float(ps[:-1]) / 1000.0
            return float(ps)
        except (ValueError, TypeError):
            return None


# ── Routing Decision ─────────────────────────────────────────────────────────


class RoutingDecision(BaseModel):
    """The visible output of a routing decision — exposes reasoning to the user."""

    model_config = {"protected_namespaces": ()}

    # Selection
    selected_model: str = Field(..., description="Name of the selected model.")
    fallback_model: Optional[str] = Field(None, description="Fallback if selected fails.")

    # Reasoning (transparent to the user)
    reason: str = Field(..., description="Human-readable explanation of why this model was chosen.")
    task_category: TaskCategory = Field(TaskCategory.GENERAL)
    required_capabilities: List[Capability] = Field(default_factory=list)

    # Expected performance
    expected_latency_ms: Optional[float] = Field(None)
    expected_tokens_per_sec: Optional[float] = Field(None)
    capability_match_score: float = Field(0.0, description="Composite capability match (0-1).")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Router confidence in this selection.")

    # Scoring breakdown
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-model final scores for transparency.",
    )

    # Metadata
    decided_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    decision_time_ms: float = Field(0.0, description="Time taken to compute routing decision.")


# ── Benchmark Schemas ────────────────────────────────────────────────────────


class BenchmarkTask(BaseModel):
    """A small standardized task for benchmarking a model locally."""

    task_id: str = Field(..., description="Unique task identifier.")
    name: str = Field(..., description="Human-readable task name.")
    category: TaskCategory = Field(TaskCategory.GENERAL)
    prompt: str = Field(..., description="The prompt to send to the model.")
    expected_contains: List[str] = Field(
        default_factory=list,
        description="Substrings the response should contain for a pass.",
    )
    max_tokens: int = Field(256, description="Maximum tokens for the response.")
    timeout_seconds: float = Field(30.0, description="Maximum time allowed.")


class BenchmarkResult(BaseModel):
    """Result of running a single benchmark task against a model."""

    model_config = {"protected_namespaces": ()}

    model_name: str
    task_id: str
    success: bool
    latency_ms: float
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    response_preview: str = Field("", description="First 200 chars of response.")
    error: Optional[str] = None
    run_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Routing Configuration ───────────────────────────────────────────────────


class CapabilityWeight(BaseModel):
    """Weight for a capability in scoring — configurable without code changes."""

    capability: Capability
    weight: float = Field(1.0, ge=0.0, le=10.0)


class TaskRule(BaseModel):
    """Configurable rule mapping a task category to required capabilities."""

    category: TaskCategory
    required_capabilities: List[Capability] = Field(default_factory=list)
    preferred_capabilities: List[Capability] = Field(default_factory=list)
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    max_parameter_size_b: Optional[float] = Field(
        None, description="Maximum model size in billions (for speed preference)."
    )
    prefer_fastest: bool = Field(
        True, description="When True, prefer faster/smaller models if capable."
    )


class RouterConfig(BaseModel):
    """Top-level routing configuration — editable as JSON without code changes."""

    # Global settings
    confidence_threshold: float = Field(
        0.4, ge=0.0, le=1.0,
        description="Minimum confidence required; below this, use fallback.",
    )
    fallback_model: Optional[str] = Field(
        None, description="Global fallback model when no candidate meets threshold.",
    )
    prefer_smallest: bool = Field(
        True, description="When capabilities are equal, prefer smaller/faster model.",
    )
    enable_benchmarks: bool = Field(
        True, description="Allow router to use benchmark history for scoring.",
    )
    benchmark_weight: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Weight given to benchmark performance vs. heuristic scores.",
    )

    # Capability weights
    capability_weights: List[CapabilityWeight] = Field(
        default_factory=lambda: [
            CapabilityWeight(capability=Capability.CODING, weight=2.0),
            CapabilityWeight(capability=Capability.REASONING, weight=1.5),
            CapabilityWeight(capability=Capability.VISION, weight=1.0),
            CapabilityWeight(capability=Capability.SPEED, weight=1.0),
            CapabilityWeight(capability=Capability.PLANNING, weight=1.5),
            CapabilityWeight(capability=Capability.CHAT, weight=0.8),
        ],
    )

    # Per-category routing rules
    task_rules: List[TaskRule] = Field(
        default_factory=lambda: [
            TaskRule(
                category=TaskCategory.CODE_GENERATION,
                required_capabilities=[Capability.CODING],
                preferred_capabilities=[Capability.REASONING, Capability.INSTRUCTION],
                min_confidence=0.5,
                prefer_fastest=False,
            ),
            TaskRule(
                category=TaskCategory.CODE_REVIEW,
                required_capabilities=[Capability.CODING, Capability.REASONING],
                min_confidence=0.5,
                prefer_fastest=False,
            ),
            TaskRule(
                category=TaskCategory.DEBUGGING,
                required_capabilities=[Capability.CODING, Capability.REASONING],
                min_confidence=0.6,
                prefer_fastest=False,
            ),
            TaskRule(
                category=TaskCategory.PLANNING,
                required_capabilities=[Capability.PLANNING, Capability.REASONING],
                preferred_capabilities=[Capability.INSTRUCTION],
                min_confidence=0.5,
            ),
            TaskRule(
                category=TaskCategory.CHAT,
                required_capabilities=[Capability.CHAT],
                prefer_fastest=True,
            ),
            TaskRule(
                category=TaskCategory.SUMMARIZATION,
                required_capabilities=[Capability.SUMMARIZATION],
                prefer_fastest=True,
            ),
            TaskRule(
                category=TaskCategory.IMAGE_ANALYSIS,
                required_capabilities=[Capability.VISION],
                min_confidence=0.7,
                prefer_fastest=False,
            ),
            TaskRule(
                category=TaskCategory.REASONING,
                required_capabilities=[Capability.REASONING],
                preferred_capabilities=[Capability.INSTRUCTION],
                min_confidence=0.6,
                prefer_fastest=False,
            ),
            TaskRule(
                category=TaskCategory.GENERAL,
                required_capabilities=[Capability.CHAT],
                prefer_fastest=True,
            ),
        ],
    )

    def get_rule(self, category: TaskCategory) -> TaskRule:
        """Get routing rule for a category, or a permissive default."""
        for rule in self.task_rules:
            if rule.category == category:
                return rule
        return TaskRule(category=category, required_capabilities=[Capability.CHAT])

    def get_capability_weight(self, cap: Capability) -> float:
        """Get weight for a capability."""
        for cw in self.capability_weights:
            if cw.capability == cap:
                return cw.weight
        return 1.0

    @classmethod
    def load_from_file(cls, path: "Path") -> "RouterConfig":
        """Load router config from JSON file."""
        import json
        from pathlib import Path as P
        p = P(path)
        if not p.is_file():
            return cls()
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.model_validate(data)
        except Exception:
            return cls()

    def save_to_file(self, path: "Path") -> None:
        """Save router config to JSON file."""
        from pathlib import Path as P
        p = P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
