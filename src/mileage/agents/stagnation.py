"""Stagnation and loop detection for local autonomous coding agents."""

from enum import Enum
import hashlib
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class StagnationReason(str, Enum):
    """Specific cause that triggered an execution loop stagnation."""

    REPEATED_ERRORS = "repeated_errors"
    UNCHANGED_EDITS = "unchanged_edits"
    MAX_RETRIES = "max_retries"
    STALLED_EXECUTION = "stalled_execution"
    OSCILLATION = "oscillation"


class StagnationReport(BaseModel):
    """Diagnostics and recommendations when stagnation is detected."""

    is_stagnant: bool
    reason: Optional[StagnationReason] = None
    message: str = ""
    details: str = ""
    threshold_value: int = 0
    current_value: int = 0
    recommendations: List[str] = Field(default_factory=list)


class StagnationDetector:
    """Monitors repeated errors, unchanged edits, excessive retries, and stalled execution.

    Stops the agent execution loop when stagnation thresholds are exceeded.
    """

    def __init__(
        self,
        repeated_error_threshold: int = 3,
        unchanged_edits_threshold: int = 2,
        max_iterations: int = 10,
        stalled_threshold: int = 3,
    ):
        self.repeated_error_threshold = repeated_error_threshold
        self.unchanged_edits_threshold = unchanged_edits_threshold
        self.max_iterations = max_iterations
        self.stalled_threshold = stalled_threshold

        # Internal tracking states
        self.current_iteration = 0
        self.last_error_signature: Optional[str] = None
        self.consecutive_repeated_errors = 0

        self.consecutive_unchanged_edits = 0
        self.total_unchanged_edits = 0

        self.consecutive_stalled_iterations = 0
        self.edits_in_current_iteration = 0

        # File state tracking for oscillation: {path: [hash1, hash2, ...]}
        self.file_history: Dict[str, List[str]] = {}

        self._stagnation_report: Optional[StagnationReport] = None

    def _normalize_error(self, error_text: str) -> str:
        """Strip volatile elements (timestamps, memory addresses, line numbers) for comparison."""
        clean = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", error_text)
        clean = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s]*", "TIMESTAMP", clean)
        clean = re.sub(r"in \d+\.\d+s", "in Xs", clean)
        # Take first 500 characters of error core
        return clean.strip()[:500]

    def record_iteration_start(self) -> None:
        """Mark the beginning of a new execution loop iteration."""
        self.current_iteration += 1
        self.edits_in_current_iteration = 0

    def record_edit(self, path: str, is_changed: bool, content: Optional[str] = None) -> Optional[StagnationReport]:
        """Record a file modification attempt and check for unchanged edits or oscillation."""
        self.edits_in_current_iteration += 1

        if not is_changed:
            self.consecutive_unchanged_edits += 1
            self.total_unchanged_edits += 1
            if self.consecutive_unchanged_edits >= self.unchanged_edits_threshold:
                self._stagnation_report = StagnationReport(
                    is_stagnant=True,
                    reason=StagnationReason.UNCHANGED_EDITS,
                    threshold_value=self.unchanged_edits_threshold,
                    current_value=self.consecutive_unchanged_edits,
                    message=(
                        f"Agent made {self.consecutive_unchanged_edits} consecutive edits "
                        f"with 0 changes to '{path}'."
                    ),
                    details="The model generated output identical to the existing file content without modifying it.",
                    recommendations=[
                        "Inspect file content and instruct the model with explicit target line replacements.",
                        "Verify that the model has the latest version of the file in context.",
                    ],
                )
                return self._stagnation_report
        else:
            self.consecutive_unchanged_edits = 0

        # Check oscillation
        if content is not None:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
            history = self.file_history.setdefault(path, [])
            if len(history) >= 2 and content_hash in history[:-1]:
                # Returned to a previous state!
                self._stagnation_report = StagnationReport(
                    is_stagnant=True,
                    reason=StagnationReason.OSCILLATION,
                    threshold_value=2,
                    current_value=len(history),
                    message=f"Agent reverted file '{path}' back to a previous state, causing an oscillation loop.",
                    details=f"File state cycle detected: hash {content_hash} previously seen.",
                    recommendations=[
                        "The agent is alternating between conflicting implementations.",
                        "Provide clearer architectural guidance or pin the interface specification.",
                    ],
                )
                return self._stagnation_report
            history.append(content_hash)

        return None

    def record_error(self, error_summary: str) -> Optional[StagnationReport]:
        """Record an error or test failure and check for repeated failures."""
        if not error_summary or not error_summary.strip():
            self.consecutive_repeated_errors = 0
            self.last_error_signature = None
            return None

        sig = hashlib.md5(self._normalize_error(error_summary).encode("utf-8")).hexdigest()

        if sig == self.last_error_signature:
            self.consecutive_repeated_errors += 1
        else:
            self.consecutive_repeated_errors = 1
            self.last_error_signature = sig

        if self.consecutive_repeated_errors >= self.repeated_error_threshold:
            short_preview = error_summary.strip().splitlines()[0][:120]
            self._stagnation_report = StagnationReport(
                is_stagnant=True,
                reason=StagnationReason.REPEATED_ERRORS,
                threshold_value=self.repeated_error_threshold,
                current_value=self.consecutive_repeated_errors,
                message=(
                    f"Same error repeated {self.consecutive_repeated_errors} times consecutively: {short_preview}"
                ),
                details=error_summary[:400],
                recommendations=[
                    "The model's repairs are failing to fix the root cause.",
                    "Review the stack trace and ensure required dependencies/types are present.",
                ],
            )
            return self._stagnation_report

        return None

    def record_iteration_end(self, tests_passed: bool) -> Optional[StagnationReport]:
        """Finalize an iteration and evaluate retry counts and stalled execution."""
        # 1. Check max retries
        if self.current_iteration >= self.max_iterations and not tests_passed:
            self._stagnation_report = StagnationReport(
                is_stagnant=True,
                reason=StagnationReason.MAX_RETRIES,
                threshold_value=self.max_iterations,
                current_value=self.current_iteration,
                message=f"Reached maximum iteration threshold ({self.max_iterations} iterations).",
                details=f"Agent executed {self.current_iteration} loops without achieving all passing tests.",
                recommendations=[
                    "Increase --max-iterations if the task is complex.",
                    "Break the task into smaller sub-tasks.",
                ],
            )
            return self._stagnation_report

        # 2. Check stalled execution (no edits made in this iteration and tests failed)
        if not tests_passed and self.edits_in_current_iteration == 0:
            self.consecutive_stalled_iterations += 1
            if self.consecutive_stalled_iterations >= self.stalled_threshold:
                self._stagnation_report = StagnationReport(
                    is_stagnant=True,
                    reason=StagnationReason.STALLED_EXECUTION,
                    threshold_value=self.stalled_threshold,
                    current_value=self.consecutive_stalled_iterations,
                    message=f"Execution stalled: {self.consecutive_stalled_iterations} iterations without any file edits.",
                    details="Agent looped without performing any concrete code modifications.",
                    recommendations=[
                        "Ensure the model is encouraged to produce concrete file write actions.",
                    ],
                )
                return self._stagnation_report
        else:
            self.consecutive_stalled_iterations = 0

        return None

    def check(self) -> StagnationReport:
        """Return the current stagnation status."""
        if self._stagnation_report:
            return self._stagnation_report
        return StagnationReport(is_stagnant=False)

    def reset(self) -> None:
        """Reset internal monitors."""
        self.current_iteration = 0
        self.last_error_signature = None
        self.consecutive_repeated_errors = 0
        self.consecutive_unchanged_edits = 0
        self.total_unchanged_edits = 0
        self.consecutive_stalled_iterations = 0
        self.edits_in_current_iteration = 0
        self.file_history.clear()
        self._stagnation_report = None
