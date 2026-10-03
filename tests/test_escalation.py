"""Tests for CopilotEscalation engine: prompt generation, VS Code opening, and logging."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mileage.agents.escalation import CopilotEscalation
from mileage.agents.evaluator_schemas import (
    AcceptanceCriterionResult,
    EscalationReason,
    EscalationRecord,
    EvaluationReport,
    RequirementResult,
    RequirementVerdict,
)
from mileage.agents.planner_schemas import (
    AcceptanceCriterion,
    ActionPlan,
    PlannerFileRef,
    Requirement,
)


def _make_plan(**kwargs):
    defaults = {
        "goal": "Implement user authentication",
        "requirements": [
            Requirement(id="R1", description="Login endpoint must exist", priority="must"),
            Requirement(id="R2", description="Password must be hashed", priority="must"),
        ],
        "acceptance_criteria": [
            AcceptanceCriterion(id="AC1", description="All auth tests pass"),
        ],
        "files": [
            PlannerFileRef(path="src/auth.py", role="target"),
            PlannerFileRef(path="tests/test_auth.py", role="context"),
        ],
    }
    defaults.update(kwargs)
    return ActionPlan(**defaults)


def _make_report(
    verdict=RequirementVerdict.FAIL,
    tests_passed=False,
    overcapacity=True,
):
    report = EvaluationReport(
        plan_id="plan_test",
        goal="Implement user authentication",
        model_name="qwen2.5-coder:7b",
        overall_verdict=verdict,
        overall_confidence=0.3,
        tests_passed=tests_passed,
        test_summary="pytest: 2 passed, 3 failed",
        test_failure_details="test_login FAILED: ConnectionError",
        overcapacity_detected=overcapacity,
        overcapacity_signals=["High fail ratio", "Stagnation detected"],
    )
    report.requirement_results = [
        RequirementResult(
            requirement_id="R1",
            description="Login endpoint must exist",
            verdict=RequirementVerdict.PASS,
            confidence=0.9,
            evidence="File src/auth.py exists with login() function",
        ),
        RequirementResult(
            requirement_id="R2",
            description="Password must be hashed",
            verdict=RequirementVerdict.FAIL,
            confidence=0.2,
            evidence="No bcrypt import found in auth.py",
        ),
    ]
    report.acceptance_results = [
        AcceptanceCriterionResult(
            criterion_id="AC1",
            description="All auth tests pass",
            verdict=RequirementVerdict.FAIL,
            confidence=0.3,
            evidence="3 tests failing",
        ),
    ]
    report.files_verified = ["src/auth.py"]
    return report


class TestCopilotPromptGeneration:
    def test_prompt_contains_goal(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        prompt = esc._build_copilot_prompt(plan, report, EscalationReason.OVERCAPACITY)

        assert "Implement user authentication" in prompt
        assert "Overcapacity" in prompt or "overcapacity" in prompt

    def test_prompt_contains_failed_requirements(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        prompt = esc._build_copilot_prompt(plan, report, EscalationReason.REPEATED_FAILURES)

        assert "R2" in prompt
        assert "Password must be hashed" in prompt
        assert "FAIL" in prompt or "🔴" in prompt

    def test_prompt_contains_test_failures(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        prompt = esc._build_copilot_prompt(plan, report, EscalationReason.OVERCAPACITY)

        assert "test_login FAILED" in prompt

    def test_prompt_contains_key_files(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        prompt = esc._build_copilot_prompt(plan, report, EscalationReason.OVERCAPACITY)

        assert "src/auth.py" in prompt

    def test_prompt_includes_session_info(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        session = MagicMock()
        session.total_iterations = 7
        session.status = MagicMock()
        session.status.value = "stagnated"
        session.files_modified = ["src/auth.py", "tests/test_auth.py"]

        prompt = esc._build_copilot_prompt(
            plan, report, EscalationReason.STAGNATION_DETECTED, session
        )

        assert "7" in prompt
        assert "stagnated" in prompt

    def test_prompt_copilot_action_section(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        prompt = esc._build_copilot_prompt(plan, report, EscalationReason.OVERCAPACITY)

        assert "What Copilot Should Do" in prompt


class TestEscalateMethod:
    @patch.object(CopilotEscalation, "_open_vscode", return_value=True)
    def test_escalate_writes_prompt_file(self, mock_vscode, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        record = esc.escalate(
            plan=plan,
            report=report,
            reason=EscalationReason.OVERCAPACITY,
            open_vscode=True,
        )

        prompt_file = tmp_path / CopilotEscalation.PROMPT_FILENAME
        assert prompt_file.is_file()
        content = prompt_file.read_text(encoding="utf-8")
        assert "Implement user authentication" in content

    @patch.object(CopilotEscalation, "_open_vscode", return_value=True)
    def test_escalate_returns_record(self, mock_vscode, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        record = esc.escalate(
            plan=plan,
            report=report,
            reason=EscalationReason.LOW_EVAL_CONFIDENCE,
        )

        assert isinstance(record, EscalationRecord)
        assert record.reason == EscalationReason.LOW_EVAL_CONFIDENCE
        assert record.goal == "Implement user authentication"
        assert record.vscode_opened is True
        assert record.copilot_prompt != ""
        assert record.escalation_id.startswith("esc_")

    @patch.object(CopilotEscalation, "_open_vscode", return_value=True)
    def test_escalate_writes_log_file(self, mock_vscode, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        esc.escalate(
            plan=plan,
            report=report,
            reason=EscalationReason.OVERCAPACITY,
        )

        log_path = tmp_path / ".mileage" / "escalation_log.jsonl"
        assert log_path.is_file()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        log_entry = json.loads(lines[0])
        assert log_entry["reason"] == "overcapacity"

    @patch.object(CopilotEscalation, "_open_vscode", return_value=False)
    def test_escalate_without_vscode(self, mock_vscode, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        record = esc.escalate(
            plan=plan,
            report=report,
            reason=EscalationReason.PARTIAL_COMPLETION,
            open_vscode=True,
        )

        assert record.vscode_opened is False

    @patch.object(CopilotEscalation, "_open_vscode")
    def test_escalate_skip_vscode(self, mock_vscode, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        record = esc.escalate(
            plan=plan,
            report=report,
            reason=EscalationReason.OVERCAPACITY,
            open_vscode=False,
        )

        mock_vscode.assert_not_called()
        assert record.vscode_opened is False

    @patch.object(CopilotEscalation, "_open_vscode", return_value=True)
    def test_escalate_with_session_record(self, mock_vscode, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        plan = _make_plan()
        report = _make_report()

        session = MagicMock()
        session.total_iterations = 10
        session.total_duration_ms = 45000.0
        session.total_tokens = 12000

        record = esc.escalate(
            plan=plan,
            report=report,
            reason=EscalationReason.STAGNATION_DETECTED,
            session_record=session,
        )

        assert record.local_attempts == 10
        assert record.local_duration_ms == 45000.0
        assert record.local_tokens_used == 12000


class TestVSCodeDetection:
    def test_find_vscode_cli_with_shutil(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)

        with patch("shutil.which", return_value="/usr/bin/code"):
            result = esc._find_vscode_cli()
            assert result == "/usr/bin/code"

    def test_find_vscode_cli_not_found(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)

        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                result = esc._find_vscode_cli()
                assert result is None


class TestReasonSummaries:
    def test_all_reasons_have_summaries(self, tmp_path):
        esc = CopilotEscalation(workspace_root=tmp_path)
        report = _make_report()

        for reason in EscalationReason:
            summary = esc._summarize_reason(reason, report)
            assert isinstance(summary, str)
            assert len(summary) > 10
