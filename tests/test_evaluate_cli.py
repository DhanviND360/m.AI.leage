"""Tests for `mileage evaluate` CLI command and the UI renderers."""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typer.testing import CliRunner

from mileage.cli import app
from mileage.agents.evaluator_schemas import (
    AcceptanceCriterionResult,
    EvaluationReport,
    RequirementResult,
    RequirementVerdict,
    EscalationRecord,
    EscalationReason,
)

runner = CliRunner()


@pytest.fixture
def plan_file(tmp_path):
    """Create a minimal ActionPlan JSON file."""
    plan = {
        "goal": "Add greeting function",
        "requirements": [
            {"id": "R1", "description": "greeting() must exist", "priority": "must"},
        ],
        "acceptance_criteria": [
            {"id": "AC1", "description": "Tests pass"},
        ],
        "files": [
            {"path": "src/greet.py", "role": "target"},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return str(plan_path)


class TestEvaluateCLI:
    def test_missing_plan_file(self):
        result = runner.invoke(app, ["evaluate", "nonexistent.json", "--no-vscode"])
        assert result.exit_code != 0

    @patch("mileage.cli.WorkspaceManager")
    @patch("mileage.cli.OllamaClient")
    def test_invalid_plan_json(self, mock_ollama_cls, mock_ws_cls, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        result = runner.invoke(app, ["evaluate", str(bad_file), "--no-vscode"])
        assert result.exit_code != 0


class TestEvaluationUI:
    """Test the UI rendering functions don't crash on various report shapes."""

    def test_render_evaluation_report_pass(self, capsys):
        from mileage.ui.components import render_evaluation_report

        report = EvaluationReport(
            plan_id="p1",
            goal="Test goal",
            overall_verdict=RequirementVerdict.PASS,
            overall_confidence=0.9,
            pass_count=3,
            partial_count=0,
            fail_count=0,
            total_requirements=3,
            tests_passed=True,
            test_summary="pytest: 3 passed",
        )
        report.requirement_results = [
            RequirementResult(
                requirement_id="R1", description="Exists",
                verdict=RequirementVerdict.PASS, confidence=0.9, evidence="found",
            ),
        ]
        # Should not raise
        render_evaluation_report(report)

    def test_render_evaluation_report_fail_with_overcapacity(self, capsys):
        from mileage.ui.components import render_evaluation_report

        report = EvaluationReport(
            plan_id="p1",
            goal="Test",
            overall_verdict=RequirementVerdict.FAIL,
            overall_confidence=0.2,
            overcapacity_detected=True,
            overcapacity_signals=["High fail ratio", "Low confidence"],
        )
        # Should not raise
        render_evaluation_report(report)

    def test_render_escalation(self, capsys):
        from mileage.ui.components import render_escalation

        record = EscalationRecord(
            reason=EscalationReason.OVERCAPACITY,
            goal="Test goal",
            model_name="test-model",
            local_attempts=5,
            vscode_opened=True,
            copilot_prompt_file="/tmp/prompt.md",
        )
        # Should not raise
        render_escalation(record)

    def test_render_escalation_no_vscode(self, capsys):
        from mileage.ui.components import render_escalation

        record = EscalationRecord(
            reason=EscalationReason.PARTIAL_COMPLETION,
            goal="Goal",
            model_name="m",
            vscode_opened=False,
            copilot_prompt_file="/tmp/p.md",
        )
        render_escalation(record)
