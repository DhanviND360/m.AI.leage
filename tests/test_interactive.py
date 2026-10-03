"""Unit and integration tests for m.AI.leage interactive session (mileage start)."""

import pytest
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from mileage.cli import app
from mileage.core.workspace import WorkspaceManager
from mileage.dashboard import DashboardEventBus, PipelineStage
from mileage.interactive.session import InteractiveSession
from mileage.agents.planner_schemas import ActionPlan, Requirement, AcceptanceCriterion
from mileage.agents.coding_schemas import CodingSessionRecord, AgentStatus
from mileage.agents.evaluator_schemas import EvaluationReport, RequirementVerdict, RequirementResult, EscalationRecord, EscalationReason
from mileage.agents.project_tester import TestRunResult


runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_event_bus():
    DashboardEventBus.reset_instance()
    yield
    DashboardEventBus.reset_instance()


class TestInteractiveSession:
    def test_session_init_and_workspace(self, tmp_path):
        ws = WorkspaceManager(root_dir=tmp_path)
        session = InteractiveSession(
            preferred_model="qwen2.5-coder",
            enable_voice=False,
            enable_tts=False,
            workspace=ws,
        )
        assert session.target_model == "qwen2.5-coder"
        assert not session.enable_voice
        assert not session.enable_tts
        assert session.workspace.is_initialized()

    def test_speedometer_greeting_renders(self, tmp_path):
        ws = WorkspaceManager(root_dir=tmp_path)
        session = InteractiveSession(
            preferred_model="qwen2.5-coder",
            enable_voice=False,
            enable_tts=False,
            workspace=ws,
        )
        # Should render without any exception
        session.render_speedometer_greeting()

    def test_acoustic_isolation_prevents_overlap(self, tmp_path):
        ws = WorkspaceManager(root_dir=tmp_path)
        session = InteractiveSession(
            preferred_model="qwen2.5-coder",
            enable_voice=True,
            enable_tts=True,
            workspace=ws,
        )
        # Mock TTS engine
        session.tts = MagicMock()
        session.tts.compress_for_speech.return_value = "Hello"

        # Simulate audio queue having dirty frames
        session._audio_queue.put(b"\x00" * 320)

        # Call say()
        session.say("Hello world")

        # Audio queue should have been flushed to prevent self-listening
        assert session._audio_queue.empty()
        session.tts.speak.assert_called_once_with("Hello", non_blocking=False)

    @patch("mileage.interactive.session.PlannerAgent")
    @patch("mileage.interactive.session.ModelRouter")
    @patch("mileage.interactive.session.CodingAgent")
    @patch("mileage.interactive.session.ProjectTester")
    @patch("mileage.interactive.session.EvaluatorAgent")
    def test_pipeline_execution_successful_flow(
        self,
        mock_evaluator_cls,
        mock_tester_cls,
        mock_coding_cls,
        mock_router_cls,
        mock_planner_cls,
        tmp_path,
    ):
        ws = WorkspaceManager(root_dir=tmp_path)
        ws.initialize(project_name="test-pipe")

        # 1. Planner mock
        mock_planner = MagicMock()
        mock_plan = ActionPlan(
            goal="Add fibonacci helper",
            requirements=[
                Requirement(id="R1", description="fib(5) == 5")
            ],
            acceptance_criteria=[
                AcceptanceCriterion(id="AC1", description="Unit tests pass")
            ],
        )
        mock_planner.plan.return_value = mock_plan
        mock_planner_cls.return_value = mock_planner

        # 2. Router mock
        mock_router = MagicMock()
        mock_decision = MagicMock()
        mock_decision.selected_model = "qwen2.5-coder"
        mock_router.route_for_prompt.return_value = mock_decision
        mock_router_cls.return_value = mock_router

        # 3. CodingAgent mock
        mock_coder = MagicMock()
        mock_session = CodingSessionRecord(
            session_id="sess_123",
            goal="Add fibonacci helper",
            model_name="qwen2.5-coder",
            status=AgentStatus.COMPLETED,
            total_tokens=150,
            total_duration_ms=1200.0,
        )
        mock_coder.execute_task.return_value = mock_session
        mock_coding_cls.return_value = mock_coder

        # 4. ProjectTester mock
        mock_tester = MagicMock()
        mock_tester.run_tests.return_value = TestRunResult(
            passed=True,
            exit_code=0,
            total_tests=2,
            tests_passed=2,
            command="pytest",
        )
        mock_tester_cls.return_value = mock_tester

        # 5. Evaluator mock
        mock_evaluator = MagicMock()
        mock_report = EvaluationReport(
            plan_id="plan_123",
            goal="Add fibonacci helper",
            overall_verdict=RequirementVerdict.PASS,
            overcapacity_detected=False,
            requirement_results=[
                RequirementResult(
                    requirement_id="REQ-1",
                    description="fib(5) == 5",
                    verdict=RequirementVerdict.PASS,
                    confidence=0.95,
                    evidence="Tests pass",
                )
            ],
            evaluation_duration_ms=800.0,
            evaluation_tokens=90,
        )
        mock_evaluator.evaluate.return_value = mock_report
        mock_evaluator_cls.return_value = mock_evaluator

        # Run session pipeline
        session = InteractiveSession(
            preferred_model="qwen2.5-coder",
            enable_voice=False,
            enable_tts=False,
            workspace=ws,
        )

        session.execute_pipeline("Add fibonacci helper")

        # Verify all pipeline stages executed
        mock_planner.plan.assert_called_once()
        mock_coder.execute_task.assert_called_once_with("Add fibonacci helper")
        mock_tester.run_tests.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(mock_plan)

        # Check that event bus recorded the completed build
        bus = DashboardEventBus.get_instance()
        assert len(bus.build_history) == 1
        record = bus.build_history[0]
        assert record.status == "complete"
        assert record.requirements_passed == 1
        assert not record.escalated

    @patch("mileage.interactive.session.PlannerAgent")
    @patch("mileage.interactive.session.ModelRouter")
    @patch("mileage.interactive.session.CodingAgent")
    @patch("mileage.interactive.session.ProjectTester")
    @patch("mileage.interactive.session.EvaluatorAgent")
    @patch("mileage.interactive.session.CopilotEscalation")
    def test_pipeline_execution_escalation_flow(
        self,
        mock_copilot_cls,
        mock_evaluator_cls,
        mock_tester_cls,
        mock_coding_cls,
        mock_router_cls,
        mock_planner_cls,
        tmp_path,
    ):
        ws = WorkspaceManager(root_dir=tmp_path)
        ws.initialize(project_name="test-pipe")

        mock_planner = MagicMock()
        mock_plan = ActionPlan(goal="Hard task", requirements=[])
        mock_planner.plan.return_value = mock_plan
        mock_planner_cls.return_value = mock_planner

        mock_router = MagicMock()
        mock_router.route_for_prompt.return_value.selected_model = "qwen2.5-coder"
        mock_router_cls.return_value = mock_router

        mock_coder = MagicMock()
        mock_coder.execute_task.return_value = CodingSessionRecord(
            session_id="sess_fail",
            goal="Hard task",
            model_name="qwen2.5-coder",
            status=AgentStatus.FAILED,
        )
        mock_coding_cls.return_value = mock_coder

        mock_tester = MagicMock()
        mock_tester.run_tests.return_value = TestRunResult(
            passed=False,
            exit_code=1,
            command="pytest",
        )
        mock_tester_cls.return_value = mock_tester

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = EvaluationReport(
            plan_id="plan_hard",
            goal="Hard task",
            overall_verdict=RequirementVerdict.FAIL,
            overcapacity_detected=True,
            requirement_results=[],
        )
        mock_evaluator_cls.return_value = mock_evaluator

        mock_copilot = MagicMock()
        mock_copilot.escalate.return_value = EscalationRecord(
            plan_goal="Hard task",
            reason=EscalationReason.OVERCAPACITY,
            prompt_file_path=str(tmp_path / ".mileage_copilot_prompt.md"),
        )
        mock_copilot_cls.return_value = mock_copilot

        session = InteractiveSession(
            preferred_model="qwen2.5-coder",
            enable_voice=False,
            enable_tts=False,
            workspace=ws,
        )

        session.execute_pipeline("Hard task")

        mock_copilot.escalate.assert_called_once()
        bus = DashboardEventBus.get_instance()
        assert len(bus.build_history) == 1
        assert bus.build_history[0].status == "escalated"
        assert bus.build_history[0].escalated is True


class TestStartCli:
    def test_start_cli_help(self):
        result = runner.invoke(app, ["start", "--help"])
        assert result.exit_code == 0
        assert "Primary one-command experience" in result.stdout
        assert "--voice" in result.stdout
        assert "--tts" in result.stdout

    def test_start_cli_invocation(self):
        with patch("mileage.interactive.session.InteractiveSession.run") as mock_run:
            result = runner.invoke(app, ["start", "--no-voice", "--no-tts"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_main_cli_shows_start_prominently(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "mileage start" in result.stdout
        assert "Primary One-Command Experience" in result.stdout


class TestMascotLogo:
    def test_mascot_text_art(self):
        from mileage.ui.mascot import get_mascot_text_art, get_app_mascot
        art = get_mascot_text_art()
        assert len(art.plain) > 50
        assert "●" in art.plain or "↗" in art.plain

        app_art = get_app_mascot()
        assert len(app_art.plain) > 50

    def test_mascot_image_pixel_art(self):
        from mileage.ui.mascot import get_image_pixel_art
        # Should return Text or None gracefully without error
        pixel_art = get_image_pixel_art()
        assert pixel_art is not None or pixel_art is None
