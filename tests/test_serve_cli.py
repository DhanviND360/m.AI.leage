"""CLI integration tests for 'mileage serve'."""

import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch

from mileage.cli import app


runner = CliRunner()


def test_serve_cli_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Command Center web dashboard" in result.stdout
    assert "--port" in result.stdout
    assert "--host" in result.stdout


def test_main_help_includes_serve():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.stdout


def test_serve_cli_invocation_mocked():
    with patch("mileage.dashboard.server.DashboardServer.start") as mock_start:
        result = runner.invoke(app, ["serve", "--port", "3999", "--host", "127.0.0.1"])
        assert result.exit_code == 0
        mock_start.assert_called_once_with(blocking=True)
