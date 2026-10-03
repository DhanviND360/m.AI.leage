"""Data models for local execution metrics and telemetry."""

from typing import Dict, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class MetricRecord(BaseModel):
    """An individual recorded execution metric event."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    command: str
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None


class MetricsSummary(BaseModel):
    """Aggregated local telemetry statistics."""

    model_config = {"protected_namespaces": ()}

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    success_rate_percent: float = 100.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    average_latency_ms: float = 0.0
    model_usage_counts: Dict[str, int] = Field(default_factory=dict)
    first_run_at: Optional[str] = None
    last_run_at: Optional[str] = None
