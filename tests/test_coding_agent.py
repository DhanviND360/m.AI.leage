"""Unit and integration tests for the autonomous CodingAgent."""

import json
from pathlib import Path
import pytest
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from mileage.agents.coding_agent import CodingAgent
from mileage.agents.coding_schemas import AgentPhase, AgentStatus
from mileage.agents.stagnation import StagnationReason
from mileage.core.workspace import WorkspaceManager
from mileage.metrics.tracker import MetricsTracker
from mileage.models.ollama_client import OllamaClient


@pytest.fixture
def workspace_with_buggy_code():
    """Create a temporary initialized workspace with a failing test."""
    temp_dir = Path(tempfile.mkdtemp())
    workspace = WorkspaceManager(temp_dir)
    workspace.initialize(project_name="calc_project")

    # Source code with a bug
    src_dir = temp_dir / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("")
    (src_dir / "calculator.py").write_text(
        "def divide(a, b):\n"
        "    # Bug: returns 0 instead of division\n"
        "    return 0\n"
    )

    # Test file that currently fails
    tests_dir = temp_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calculator.py").write_text(
        "from src.calculator import divide\n\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5\n"
    )

    yield workspace

    shutil.rmtree(temp_dir, ignore_errors=True)


class MockOllamaClient:
    """Mock client returning sequential predefined responses for testing."""

    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        self.host = "http://127.0.0.1:11434"

    def start_daemon(self, timeout_seconds: float = 5.0) -> bool:
        return True

    def check_health(self, quick_timeout: float = 1.0):
        m = MagicMock()
        m.is_running = True
        return m

    def resolve_active_model(self, preferred_model=None):
        return preferred_model or "mock-model"

    def _get_http_client(self, timeout: float = 60.0):
        # Mock httpx context manager
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        # Return next response
        if self.call_count < len(self.responses):
            resp_content = self.responses[self.call_count]
        else:
            resp_content = self.responses[-1]
        self.call_count += 1

        mock_resp.json.return_value = {
            "message": {"content": resp_content},
            "prompt_eval_count": 150,
            "eval_count": 80,
        }

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        return mock_client


class TestCodingAgent:
    """Test autonomous CodingAgent execution loop, repairs, stagnation, and metrics."""

    def test_inspect_plan_edit_test_repair_success(self, workspace_with_buggy_code):
        """Test loop: Iteration 1 makes edit that still fails -> Iteration 2 repairs and passes."""
        # Response 1: Attempts a bad fix (returns 4 instead of 5)
        resp1 = json.dumps({
            "thought": "I will fix the divide function.",
            "phase": "edit",
            "summary": "Fix divide function",
            "actions": [
                {
                    "tool": "write_file",
                    "args": {
                        "relative_path": "src/calculator.py",
                        "content": "def divide(a, b):\n    return 4\n",
                    },
                }
            ],
        })

        # Response 2: Receives test failure and applies correct fix
        resp2 = json.dumps({
            "thought": "Test failed with divide(10, 2) == 5. I will return a // b.",
            "phase": "complete",
            "summary": "Fixed divide function to return a // b",
            "actions": [
                {
                    "tool": "write_file",
                    "args": {
                        "relative_path": "src/calculator.py",
                        "content": "def divide(a, b):\n    return a // b\n",
                    },
                }
            ],
        })

        mock_client = MockOllamaClient([resp1, resp2])
        tracker = MetricsTracker(workspace_with_buggy_code.metrics_path)

        events_received = []

        def on_event(ev):
            events_received.append(ev)

        agent = CodingAgent(
            ollama_client=mock_client,
            model_name="mock-model",
            workspace=workspace_with_buggy_code,
            metrics_tracker=tracker,
            max_iterations=5,
            auto_test=True,
            on_event=on_event,
        )

        session = agent.execute_task("Fix divide function in src/calculator.py so tests pass")

        # Assertions
        assert session.status == AgentStatus.COMPLETED
        assert session.tests_passed is True
        assert session.total_iterations == 2
        assert "src/calculator.py" in session.files_modified
        assert session.total_prompt_tokens > 0
        assert session.total_completion_tokens > 0
        assert session.total_tokens == session.total_prompt_tokens + session.total_completion_tokens
        assert session.total_duration_ms > 0

        # Verify file content on disk is now correct
        fixed_content = (workspace_with_buggy_code.root_dir / "src" / "calculator.py").read_text()
        assert "return a // b" in fixed_content

        # Verify session file was persisted to disk
        session_file = workspace_with_buggy_code.mileage_dir / "sessions" / f"{session.session_id}.json"
        assert session_file.is_file()

        # Verify metrics record was saved
        summary = tracker.get_summary()
        assert summary.total_runs >= 1

        # Verify events were emitted
        event_types = [e.event_type for e in events_received]
        assert "session_start" in event_types
        assert "edit" in event_types
        assert "test" in event_types
        assert "repair" in event_types
        assert "complete" in event_types

    def test_stagnation_halts_on_unchanged_edits(self, workspace_with_buggy_code):
        """Test that agent halts with STAGNATED when model produces unchanged edits."""
        # Response repeatedly writes identical content that produces no diff
        resp = json.dumps({
            "thought": "Writing the same content again",
            "phase": "edit",
            "summary": "No change",
            "actions": [
                {
                    "tool": "write_file",
                    "args": {
                        "relative_path": "src/calculator.py",
                        "content": "def divide(a, b):\n    # Bug: returns 0 instead of division\n    return 0\n",
                    },
                }
            ],
        })

        mock_client = MockOllamaClient([resp, resp, resp])
        agent = CodingAgent(
            ollama_client=mock_client,
            model_name="mock-model",
            workspace=workspace_with_buggy_code,
            unchanged_edits_threshold=2,
            max_iterations=5,
            auto_test=False,
        )

        session = agent.execute_task("Try fixing something")
        assert session.status == AgentStatus.STAGNATED
        assert session.stagnation_report is not None
        assert session.stagnation_report.reason == StagnationReason.UNCHANGED_EDITS
        assert "0 changes" in session.stagnation_report.message

    def test_stagnation_halts_on_repeated_errors(self, workspace_with_buggy_code):
        """Test that agent halts with STAGNATED when identical test failure repeats."""
        # Edit file with something that fails the exact same way every time
        resp = json.dumps({
            "thought": "Failing attempt",
            "phase": "edit",
            "summary": "Still broken",
            "actions": [
                {
                    "tool": "write_file",
                    "args": {
                        "relative_path": "src/calculator.py",
                        "content": "def divide(a, b):\n    return 999  # wrong\n",
                    },
                }
            ],
        })

        mock_client = MockOllamaClient([resp, resp, resp, resp])
        agent = CodingAgent(
            ollama_client=mock_client,
            model_name="mock-model",
            workspace=workspace_with_buggy_code,
            repeated_error_threshold=2,
            max_iterations=5,
            auto_test=True,
        )

        session = agent.execute_task("Fix divide")
        assert session.status == AgentStatus.STAGNATED
        assert session.stagnation_report is not None
        assert session.stagnation_report.reason == StagnationReason.REPEATED_ERRORS
        assert session.stagnation_report.threshold_value == 2

    def test_dry_run_does_not_modify_files(self, workspace_with_buggy_code):
        """Test that dry_run flag prevents file modifications."""
        resp = json.dumps({
            "thought": "I will rewrite calculator.py",
            "phase": "complete",
            "summary": "Done",
            "actions": [
                {
                    "tool": "write_file",
                    "args": {
                        "relative_path": "src/calculator.py",
                        "content": "def divide(a, b): return a // b\n",
                    },
                }
            ],
        })

        original_content = (workspace_with_buggy_code.root_dir / "src" / "calculator.py").read_text()

        mock_client = MockOllamaClient([resp])
        agent = CodingAgent(
            ollama_client=mock_client,
            model_name="mock-model",
            workspace=workspace_with_buggy_code,
            dry_run=True,
            auto_test=False,
        )

        session = agent.execute_task("Fix divide in dry run")
        current_content = (workspace_with_buggy_code.root_dir / "src" / "calculator.py").read_text()
        assert current_content == original_content
