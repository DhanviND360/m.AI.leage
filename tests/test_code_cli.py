"""Unit tests for the 'mileage code' CLI command."""

from pathlib import Path
import pytest
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from mileage.agents.coding_schemas import AgentStatus, CodingSessionRecord
from mileage.cli import app
from mileage.core.workspace import WorkspaceManager

runner = CliRunner()


@pytest.fixture
def clean_workspace():
    temp_dir = Path(tempfile.mkdtemp())
    ws = WorkspaceManager(temp_dir)
    ws.initialize(project_name="cli_test_proj")
    yield ws
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCodeCLI:
    """Test CLI commands for mileage code and mileage agent."""

    def test_code_help(self):
        result = runner.invoke(app, ["code", "--help"])
        assert result.exit_code == 0
        assert "Autonomous local coding agent" in result.stdout
        assert "--max-iterations" in result.stdout
        assert "--auto-test" in result.stdout

    def test_agent_alias_help(self):
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        assert "Alias for 'mileage code'" in result.stdout

    @patch("mileage.cli.OllamaClient")
    @patch("mileage.cli.CodingAgent")
    def test_code_cmd_execution(self, mock_agent_cls, mock_ollama_cls, clean_workspace):
        # Mock ollama ensure_ready
        mock_ollama = MagicMock()
        mock_ollama.ensure_ready.return_value = (True, "test-model")
        mock_ollama_cls.return_value = mock_ollama

        # Mock coding agent
        mock_agent = MagicMock()
        mock_session = CodingSessionRecord(
            session_id="test_sess_123",
            goal="Add multiplication feature",
            model_name="test-model",
            status=AgentStatus.COMPLETED,
            total_iterations=1,
            total_duration_ms=450.0,
            total_tokens=200,
            tests_passed=True,
            test_summary="pytest: 1 passed",
            files_modified=["src/math.py"],
            final_summary="Multiplication added and verified.",
        )
        mock_agent.execute_task.return_value = mock_session
        mock_agent_cls.return_value = mock_agent

        with patch("mileage.core.workspace.WorkspaceManager.find_workspace", return_value=clean_workspace):
            result = runner.invoke(app, ["code", "Add multiplication feature"])
            assert result.exit_code == 0
            assert "Goal:" in result.stdout
            assert "Status:" in result.stdout
            assert "COMPLETED" in result.stdout
