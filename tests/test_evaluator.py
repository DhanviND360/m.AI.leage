"""Tests for EvaluatorAgent, EvaluationReport schemas, and overcapacity detection."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from mileage.agents.evaluator_schemas import (
    AcceptanceCriterionResult,
    EscalationReason,
    EscalationRecord,
    EvaluationReport,
    RequirementResult,
    RequirementVerdict,
)
from mileage.agents.evaluator import EvaluatorAgent
from mileage.agents.planner_schemas import (
    AcceptanceCriterion,
    ActionPlan,
    PlannerFileRef,
    Requirement,
)
from mileage.agents.project_tester import TestRunResult


# ── Schema Tests ─────────────────────────────────────────────────────────

class TestRequirementVerdict:
    def test_enum_values(self):
        assert RequirementVerdict.PASS == "pass"
        assert RequirementVerdict.PARTIAL == "partial"
        assert RequirementVerdict.FAIL == "fail"


class TestEvaluationReport:
    def _make_report(self, req_verdicts=None, ac_verdicts=None):
        report = EvaluationReport(plan_id="p1", goal="test goal")
        for rv in (req_verdicts or []):
            report.requirement_results.append(
                RequirementResult(
                    requirement_id=rv[0],
                    description=f"desc {rv[0]}",
                    priority=rv[2] if len(rv) > 2 else "must",
                    verdict=rv[1],
                    confidence=0.8,
                    evidence="evidence",
                )
            )
        for av in (ac_verdicts or []):
            report.acceptance_results.append(
                AcceptanceCriterionResult(
                    criterion_id=av[0],
                    description=f"desc {av[0]}",
                    verdict=av[1],
                    confidence=0.7,
                    evidence="evidence",
                )
            )
        report.compute_aggregates()
        return report

    def test_all_pass(self):
        report = self._make_report(
            [("R1", RequirementVerdict.PASS), ("R2", RequirementVerdict.PASS)],
            [("AC1", RequirementVerdict.PASS)],
        )
        assert report.overall_verdict == RequirementVerdict.PASS
        assert report.pass_count == 3
        assert report.fail_count == 0
        assert report.partial_count == 0

    def test_any_must_fail_is_fail(self):
        report = self._make_report(
            [
                ("R1", RequirementVerdict.PASS, "must"),
                ("R2", RequirementVerdict.FAIL, "must"),
            ]
        )
        assert report.overall_verdict == RequirementVerdict.FAIL

    def test_partial_only(self):
        report = self._make_report(
            [("R1", RequirementVerdict.PARTIAL, "should")],
            [("AC1", RequirementVerdict.PASS)],
        )
        assert report.overall_verdict == RequirementVerdict.PARTIAL

    def test_empty_report_is_fail(self):
        report = EvaluationReport(plan_id="p1", goal="test")
        report.compute_aggregates()
        assert report.overall_verdict == RequirementVerdict.FAIL
        assert report.overall_confidence == 0.0

    def test_confidence_average(self):
        report = EvaluationReport(plan_id="p1", goal="test")
        report.requirement_results.append(
            RequirementResult(
                requirement_id="R1", description="d", verdict=RequirementVerdict.PASS,
                confidence=0.9, evidence="e",
            )
        )
        report.requirement_results.append(
            RequirementResult(
                requirement_id="R2", description="d", verdict=RequirementVerdict.PASS,
                confidence=0.5, evidence="e",
            )
        )
        report.compute_aggregates()
        assert report.overall_confidence == 0.7  # (0.9 + 0.5) / 2

    def test_should_fail_with_must_pass_is_partial(self):
        """If a 'should' requirement fails but all 'must' pass, still not PASS."""
        report = self._make_report(
            [
                ("R1", RequirementVerdict.PASS, "must"),
                ("R2", RequirementVerdict.FAIL, "should"),
            ]
        )
        # has a fail, but no must fails → still overall FAIL because fail_count > 0
        assert report.overall_verdict == RequirementVerdict.FAIL


class TestEscalationRecord:
    def test_creation(self):
        rec = EscalationRecord(
            reason=EscalationReason.OVERCAPACITY,
            goal="Test goal",
            plan_id="p1",
        )
        assert rec.reason == EscalationReason.OVERCAPACITY
        assert rec.goal == "Test goal"
        assert rec.vscode_opened is False

    def test_serialization_roundtrip(self):
        rec = EscalationRecord(
            reason=EscalationReason.LOW_EVAL_CONFIDENCE,
            goal="Test",
            copilot_prompt="Fix the bug",
        )
        json_str = rec.model_dump_json()
        restored = EscalationRecord.model_validate_json(json_str)
        assert restored.reason == EscalationReason.LOW_EVAL_CONFIDENCE
        assert restored.copilot_prompt == "Fix the bug"


# ── EvaluatorAgent Tests ─────────────────────────────────────────────────

class TestEvaluatorAgent:
    def _make_plan(self, **kwargs):
        defaults = {
            "goal": "Add a hello world function",
            "requirements": [
                Requirement(id="R1", description="Function must exist", priority="must"),
                Requirement(id="R2", description="Function must return string", priority="should"),
            ],
            "acceptance_criteria": [
                AcceptanceCriterion(id="AC1", description="Tests pass"),
            ],
            "files": [
                PlannerFileRef(path="src/hello.py", role="target"),
            ],
        }
        defaults.update(kwargs)
        return ActionPlan(**defaults)

    def _make_evaluator(self, workspace_root=None):
        mock_client = MagicMock()
        mock_client.start_daemon = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.root_dir = Path(workspace_root or "/tmp/test_workspace")
        return EvaluatorAgent(
            ollama_client=mock_client,
            model_name="qwen2.5-coder:7b",
            workspace=mock_workspace,
        )

    @patch.object(EvaluatorAgent, "_run_tests")
    @patch.object(EvaluatorAgent, "_llm_evaluate")
    def test_evaluate_all_pass(self, mock_llm, mock_tests, tmp_path):
        # Setup: tests pass, target file exists
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "hello.py").write_text("def hello(): return 'hi'")

        mock_tests.return_value = TestRunResult(
            passed=True, exit_code=0, command="pytest",
            tests_passed=5, tests_failed=0, total_tests=5,
        )

        evaluator = self._make_evaluator(str(tmp_path))
        evaluator.workspace.root_dir = tmp_path
        plan = self._make_plan()

        report = evaluator.evaluate(plan)

        assert report.tests_passed is True
        assert report.overall_verdict == RequirementVerdict.PASS
        assert report.pass_count >= 1
        assert "src/hello.py" in report.files_verified

    @patch.object(EvaluatorAgent, "_run_tests")
    @patch.object(EvaluatorAgent, "_llm_evaluate")
    def test_evaluate_tests_fail(self, mock_llm, mock_tests, tmp_path):
        mock_tests.return_value = TestRunResult(
            passed=False, exit_code=1, command="pytest",
            tests_passed=2, tests_failed=3, total_tests=5,
            failure_summary="test_hello FAILED: AssertionError",
        )

        evaluator = self._make_evaluator(str(tmp_path))
        evaluator.workspace.root_dir = tmp_path
        plan = self._make_plan()

        report = evaluator.evaluate(plan)

        assert report.tests_passed is False
        assert report.test_failure_details != ""

    @patch.object(EvaluatorAgent, "_run_tests")
    def test_overcapacity_detection(self, mock_tests, tmp_path):
        mock_tests.return_value = TestRunResult(
            passed=False, exit_code=1, command="pytest",
            tests_passed=0, tests_failed=5, total_tests=5,
            failure_summary="All tests failed",
        )

        evaluator = self._make_evaluator(str(tmp_path))
        evaluator.workspace.root_dir = tmp_path
        evaluator.low_confidence_threshold = 0.01  # Don't trigger LLM eval

        plan = self._make_plan()

        # Mock session record with stagnation
        session = MagicMock()
        session.total_iterations = 8
        session.stagnation_report = MagicMock()
        session.stagnation_report.is_stagnant = True
        session.stagnation_report.reason = "repeated_errors"

        report = evaluator.evaluate(plan, session_record=session)

        assert report.overcapacity_detected is True
        assert len(report.overcapacity_signals) >= 2

    @patch.object(EvaluatorAgent, "_run_tests")
    def test_file_verification(self, mock_tests, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "hello.py").write_text("code")
        (tmp_path / "src" / "missing.py")  # does NOT exist

        mock_tests.return_value = TestRunResult(
            passed=True, exit_code=0, command="pytest",
            tests_passed=1, tests_failed=0, total_tests=1,
        )

        evaluator = self._make_evaluator(str(tmp_path))
        evaluator.workspace.root_dir = tmp_path

        plan = self._make_plan(files=[
            PlannerFileRef(path="src/hello.py", role="target"),
            PlannerFileRef(path="src/missing.py", role="target"),
        ])

        report = evaluator.evaluate(plan)
        assert "src/hello.py" in report.files_verified
        assert "src/missing.py" not in report.files_verified


class TestEvaluatorJsonExtraction:
    def _get_evaluator(self):
        mock_client = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.root_dir = Path("/tmp")
        return EvaluatorAgent(
            ollama_client=mock_client, model_name="test", workspace=mock_workspace
        )

    def test_plain_json(self):
        ev = self._get_evaluator()
        result = ev._extract_json('{"requirements": []}')
        assert result == {"requirements": []}

    def test_json_in_code_block(self):
        ev = self._get_evaluator()
        raw = '```json\n{"requirements": [{"requirement_id": "R1"}]}\n```'
        result = ev._extract_json(raw)
        assert result["requirements"][0]["requirement_id"] == "R1"

    def test_json_with_prefix_text(self):
        ev = self._get_evaluator()
        raw = 'Here is my analysis:\n{"requirements": []}'
        result = ev._extract_json(raw)
        assert result == {"requirements": []}

    def test_empty_returns_empty(self):
        ev = self._get_evaluator()
        assert ev._extract_json("") == {}
        assert ev._extract_json("no json here") == {}


class TestOvercapacityDetection:
    def _make_evaluator(self):
        mock_client = MagicMock()
        mock_workspace = MagicMock()
        mock_workspace.root_dir = Path("/tmp")
        return EvaluatorAgent(
            ollama_client=mock_client, model_name="test", workspace=mock_workspace
        )

    def test_no_signals_no_overcapacity(self):
        ev = self._make_evaluator()
        report = EvaluationReport(plan_id="p1", goal="test")
        report.tests_passed = True
        report.overall_confidence = 0.8
        report.total_requirements = 3
        report.fail_count = 0
        ev._detect_overcapacity(report, None)
        assert not report.overcapacity_detected

    def test_high_fail_ratio_signal(self):
        ev = self._make_evaluator()
        report = EvaluationReport(plan_id="p1", goal="test")
        report.total_requirements = 4
        report.fail_count = 3
        report.overall_confidence = 0.1
        ev._detect_overcapacity(report, None)
        assert report.overcapacity_detected is True
        assert any("fail ratio" in s.lower() for s in report.overcapacity_signals)

    def test_stagnation_session_signal(self):
        ev = self._make_evaluator()
        report = EvaluationReport(plan_id="p1", goal="test")
        report.total_requirements = 2
        report.fail_count = 1
        report.overall_confidence = 0.1
        report.tests_passed = False
        report.test_failure_details = "some failure"

        session = MagicMock()
        session.total_iterations = 8
        session.stagnation_report = MagicMock()
        session.stagnation_report.is_stagnant = True
        session.stagnation_report.reason = "oscillation"

        ev._detect_overcapacity(report, session)
        assert report.overcapacity_detected is True
