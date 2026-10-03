"""Copilot Escalation Engine: generates compact context-rich prompts and opens VS Code.

When the EvaluatorAgent detects overcapacity, this module:
1. Compresses the evaluation report, stagnation data, and partial progress
   into a compact Copilot-ready prompt.
2. Writes the prompt to a discoverable file in the workspace.
3. Opens the project in VS Code using the `code` CLI.
4. Optionally copies the prompt to clipboard for immediate Copilot use.
"""

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from mileage.agents.evaluator_schemas import (
    EscalationReason,
    EscalationRecord,
    EvaluationReport,
    RequirementVerdict,
)
from mileage.agents.planner_schemas import ActionPlan
from mileage.core.logger import logger


class CopilotEscalation:
    """Generates a compact, context-rich Copilot prompt and manages VS Code handoff."""

    PROMPT_FILENAME = ".mileage_copilot_prompt.md"

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()

    def escalate(
        self,
        plan: ActionPlan,
        report: EvaluationReport,
        reason: EscalationReason,
        session_record=None,
        open_vscode: bool = True,
    ) -> EscalationRecord:
        """Execute full escalation: generate prompt, write file, open VS Code."""
        esc_id = f"esc_{uuid.uuid4().hex[:8]}"
        start_t = time.perf_counter()

        # Build the compact prompt
        prompt = self._build_copilot_prompt(plan, report, reason, session_record)

        # Write prompt file to workspace
        prompt_file = self.workspace_root / self.PROMPT_FILENAME
        prompt_file.write_text(prompt, encoding="utf-8")
        logger.info("Copilot prompt written to %s", prompt_file)

        # Open VS Code
        vscode_opened = False
        if open_vscode:
            vscode_opened = self._open_vscode(prompt_file)

        # Build record
        record = EscalationRecord(
            escalation_id=esc_id,
            reason=reason,
            reason_detail=self._summarize_reason(reason, report),
            plan_id=plan.plan_id,
            goal=plan.goal,
            model_name=report.model_name,
            local_attempts=getattr(session_record, "total_iterations", 0) if session_record else 0,
            local_duration_ms=getattr(session_record, "total_duration_ms", 0.0) if session_record else 0.0,
            local_tokens_used=getattr(session_record, "total_tokens", 0) if session_record else 0,
            evaluation_report=report,
            copilot_prompt=prompt,
            copilot_prompt_file=str(prompt_file),
            vscode_opened=vscode_opened,
            workspace_path=str(self.workspace_root),
        )

        # Write escalation log
        log_path = self.workspace_root / ".mileage" / "escalation_log.jsonl"
        self._append_log(log_path, record)

        return record

    def _build_copilot_prompt(
        self,
        plan: ActionPlan,
        report: EvaluationReport,
        reason: EscalationReason,
        session_record=None,
    ) -> str:
        """Generate a compact, context-rich prompt optimized for Copilot Chat."""
        sections = []

        # Header
        sections.append(
            "# 🚀 m.AI.leage → Copilot Escalation\n"
            f"**Goal:** {plan.goal}\n"
            f"**Escalation Reason:** {reason.value}\n"
            f"**Local Model:** {report.model_name}\n"
        )

        # What was attempted
        if session_record:
            iters = getattr(session_record, "total_iterations", 0)
            status = getattr(session_record, "status", "unknown")
            if hasattr(status, "value"):
                status = status.value
            sections.append(
                f"## Local Attempt Summary\n"
                f"- Iterations: {iters}\n"
                f"- Status: {status}\n"
                f"- Files modified: {', '.join(getattr(session_record, 'files_modified', []))}\n"
            )

        # Failed requirements
        failed_reqs = [
            r for r in report.requirement_results
            if r.verdict != RequirementVerdict.PASS
        ]
        if failed_reqs:
            sections.append("## Unresolved Requirements")
            for r in failed_reqs:
                badge = "🟡 PARTIAL" if r.verdict == RequirementVerdict.PARTIAL else "🔴 FAIL"
                sections.append(
                    f"- **{r.requirement_id}** [{badge}]: {r.description}\n"
                    f"  Evidence: {r.evidence}"
                )
            sections.append("")

        # Failed acceptance criteria
        failed_acs = [
            a for a in report.acceptance_results
            if a.verdict != RequirementVerdict.PASS
        ]
        if failed_acs:
            sections.append("## Unresolved Acceptance Criteria")
            for a in failed_acs:
                badge = "🟡 PARTIAL" if a.verdict == RequirementVerdict.PARTIAL else "🔴 FAIL"
                sections.append(
                    f"- **{a.criterion_id}** [{badge}]: {a.description}\n"
                    f"  Evidence: {a.evidence}"
                )
            sections.append("")

        # Test failures
        if report.test_failure_details:
            sections.append(
                f"## Test Failures\n```\n{report.test_failure_details[:2000]}\n```\n"
            )

        # Key files
        if report.files_verified:
            sections.append(
                "## Key Project Files\n"
                + "\n".join(f"- `{f}`" for f in report.files_verified[:10])
                + "\n"
            )

        # Action needed
        sections.append(
            "## What Copilot Should Do\n"
            "1. Review the failed requirements above.\n"
            "2. Inspect the key files listed.\n"
            "3. Implement fixes for each unresolved requirement.\n"
            "4. Run the project tests to confirm all pass.\n"
        )

        return "\n".join(sections)

    def _open_vscode(self, prompt_file: Path) -> bool:
        """Open the workspace and prompt file in VS Code using the `code` CLI."""
        code_cmd = self._find_vscode_cli()
        if not code_cmd:
            logger.warning("VS Code 'code' CLI not found in PATH")
            return False

        try:
            # Open workspace folder
            subprocess.Popen(
                [code_cmd, str(self.workspace_root)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Open the prompt file directly
            subprocess.Popen(
                [code_cmd, str(prompt_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("VS Code opened for %s", self.workspace_root)
            return True
        except Exception as e:
            logger.warning("Failed to open VS Code: %s", e)
            return False

    def _find_vscode_cli(self) -> Optional[str]:
        """Locate the VS Code `code` CLI binary."""
        # Check PATH first
        code_path = shutil.which("code")
        if code_path:
            return code_path

        # Platform-specific fallbacks
        import platform
        system = platform.system()
        candidates = []
        if system == "Windows":
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            if local_app_data:
                candidates.append(
                    os.path.join(local_app_data, "Programs", "Microsoft VS Code", "bin", "code.cmd")
                )
            program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
            candidates.append(
                os.path.join(program_files, "Microsoft VS Code", "bin", "code.cmd")
            )
        elif system == "Darwin":
            candidates.append("/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code")
        else:
            candidates.extend(["/usr/bin/code", "/snap/bin/code"])

        for c in candidates:
            if os.path.isfile(c):
                return c

        return None

    def _summarize_reason(self, reason: EscalationReason, report: EvaluationReport) -> str:
        """Create a human-readable reason summary."""
        summaries = {
            EscalationReason.REPEATED_FAILURES: (
                f"Local agent produced repeated failures. "
                f"Final state: {report.fail_count} failed, {report.pass_count} passed."
            ),
            EscalationReason.LOW_EVAL_CONFIDENCE: (
                f"Evaluator confidence too low ({report.overall_confidence:.2f}). "
                f"Cannot reliably confirm implementation quality."
            ),
            EscalationReason.STAGNATION_DETECTED: (
                f"Coding agent stagnated during execution. "
                f"Overcapacity signals: {', '.join(report.overcapacity_signals[:3])}"
            ),
            EscalationReason.PARTIAL_COMPLETION: (
                f"Implementation partially complete: "
                f"{report.pass_count} pass, {report.partial_count} partial, {report.fail_count} fail."
            ),
            EscalationReason.OVERCAPACITY: (
                f"Local model overcapacity detected with {len(report.overcapacity_signals)} signals: "
                f"{'; '.join(report.overcapacity_signals[:3])}"
            ),
        }
        return summaries.get(reason, f"Escalation due to: {reason.value}")

    def _append_log(self, log_path: Path, record: EscalationRecord) -> None:
        """Append escalation record to JSONL log."""
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception as e:
            logger.warning("Failed to write escalation log: %s", e)
