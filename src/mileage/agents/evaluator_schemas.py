"""Pydantic schemas for the Evaluator Agent and Copilot escalation system."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RequirementVerdict(str, Enum):
    """Verdict for a single requirement or acceptance criterion."""

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


class EscalationReason(str, Enum):
    """Why local execution was escalated to Copilot."""

    REPEATED_FAILURES = "repeated_failures"
    LOW_EVAL_CONFIDENCE = "low_eval_confidence"
    STAGNATION_DETECTED = "stagnation_detected"
    PARTIAL_COMPLETION = "partial_completion"
    OVERCAPACITY = "overcapacity"


class RequirementResult(BaseModel):
    """Evaluation of a single requirement against the implementation."""

    requirement_id: str
    description: str
    priority: str = "must"
    verdict: RequirementVerdict
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model's confidence in this verdict (0.0 = no confidence, 1.0 = fully certain).",
    )
    evidence: str = Field(
        "", description="Concrete proof: test output, file content, or LLM-verified reasoning."
    )
    verification_method: str = Field(
        "", description="How this was verified: test_result, file_check, llm_eval, build_check."
    )


class AcceptanceCriterionResult(BaseModel):
    """Evaluation of a single acceptance criterion."""

    criterion_id: str
    description: str
    verdict: RequirementVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""
    verification_method: str = ""


class EvaluationReport(BaseModel):
    """Complete structured evaluation of implementation against an ActionPlan."""

    model_config = {"protected_namespaces": ()}

    plan_id: str
    goal: str
    model_name: str = ""
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Per-requirement verdicts
    requirement_results: List[RequirementResult] = Field(default_factory=list)
    acceptance_results: List[AcceptanceCriterionResult] = Field(default_factory=list)

    # Aggregate
    overall_verdict: RequirementVerdict = RequirementVerdict.FAIL
    overall_confidence: float = 0.0
    pass_count: int = 0
    partial_count: int = 0
    fail_count: int = 0
    total_requirements: int = 0

    # Test evidence
    tests_passed: bool = False
    test_summary: str = ""
    test_failure_details: str = ""

    # Build evidence
    build_passed: Optional[bool] = None
    build_output: str = ""

    # Files inspected
    files_verified: List[str] = Field(default_factory=list)

    # LLM evaluation (targeted model assessment)
    llm_evaluation_used: bool = False
    llm_evaluation_summary: str = ""

    # Overcapacity detection
    overcapacity_detected: bool = False
    overcapacity_signals: List[str] = Field(default_factory=list)

    # Timing
    evaluation_duration_ms: float = 0.0
    evaluation_tokens: int = 0

    def compute_aggregates(self) -> None:
        """Recompute pass/fail/partial counts and overall verdict from results."""
        all_results = list(self.requirement_results) + list(self.acceptance_results)
        self.total_requirements = len(all_results)
        self.pass_count = sum(1 for r in all_results if r.verdict == RequirementVerdict.PASS)
        self.partial_count = sum(1 for r in all_results if r.verdict == RequirementVerdict.PARTIAL)
        self.fail_count = sum(1 for r in all_results if r.verdict == RequirementVerdict.FAIL)

        if self.total_requirements == 0:
            self.overall_verdict = RequirementVerdict.FAIL
            self.overall_confidence = 0.0
            return

        # Compute average confidence
        self.overall_confidence = round(
            sum(r.confidence for r in all_results) / len(all_results), 3
        )

        # Determine aggregate verdict
        must_results = [r for r in self.requirement_results if r.priority == "must"]
        must_fails = sum(1 for r in must_results if r.verdict == RequirementVerdict.FAIL)

        if must_fails > 0:
            self.overall_verdict = RequirementVerdict.FAIL
        elif self.fail_count == 0 and self.partial_count == 0:
            self.overall_verdict = RequirementVerdict.PASS
        elif self.fail_count == 0:
            self.overall_verdict = RequirementVerdict.PARTIAL
        else:
            self.overall_verdict = RequirementVerdict.FAIL


class EscalationRecord(BaseModel):
    """Record of a local-to-Copilot escalation event."""

    model_config = {"protected_namespaces": ()}

    escalation_id: str = ""
    escalated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reason: EscalationReason
    reason_detail: str = ""

    # Original context
    plan_id: str = ""
    goal: str = ""
    model_name: str = ""
    local_attempts: int = 0
    local_duration_ms: float = 0.0
    local_tokens_used: int = 0

    # Evaluation evidence
    evaluation_report: Optional[EvaluationReport] = None

    # Generated escalation prompt
    copilot_prompt: str = ""
    copilot_prompt_file: str = ""

    # VS Code integration
    vscode_opened: bool = False
    workspace_path: str = ""
