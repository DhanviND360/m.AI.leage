"""Tests for Typer CLI commands."""

from pathlib import Path
from typer.testing import CliRunner
from mileage.cli import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output
    assert "init" in result.output
    assert "workspace" in result.output


def test_cli_init_and_status(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    init_res = runner.invoke(app, ["init", "--name", "cli-test-app"])
    assert init_res.exit_code == 0
    assert "Workspace Ready" in init_res.output

    status_res = runner.invoke(app, ["status"])
    assert status_res.exit_code == 0
    assert "cli-test-app" in status_res.output
    assert "INITIALIZED" in status_res.output


def test_cli_workspace_and_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "ws-app"])

    (tmp_path / "index.py").write_text("print('test')", encoding="utf-8")

    ws_res = runner.invoke(app, ["workspace"])
    assert ws_res.exit_code == 0
    assert "ws-app" in ws_res.output

    metrics_res = runner.invoke(app, ["metrics"])
    assert metrics_res.exit_code == 0
