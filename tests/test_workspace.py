"""Tests for WorkspaceManager and file scanning."""

from pathlib import Path
import pytest
from mileage.core.exceptions import (
    WorkspaceAlreadyInitializedError,
    WorkspaceNotInitializedError,
)
from mileage.core.workspace import WorkspaceManager


def test_workspace_initialization(tmp_path: Path):
    manager = WorkspaceManager(tmp_path)
    assert not manager.is_initialized()

    config = manager.initialize(project_name="my-cool-project")
    assert manager.is_initialized()
    assert config.workspace.project_name == "my-cool-project"
    assert (tmp_path / ".mileage" / "config.json").exists()
    assert (tmp_path / ".mileage" / "manifest.json").exists()
    assert (tmp_path / ".mileage" / "metrics.jsonl").exists()

    # Re-initialization without force should raise WorkspaceAlreadyInitializedError
    with pytest.raises(WorkspaceAlreadyInitializedError):
        manager.initialize(project_name="again")


def test_workspace_file_scan_and_stats(tmp_path: Path):
    manager = WorkspaceManager(tmp_path)
    manager.initialize(project_name="stats-project")

    # Create dummy files
    (tmp_path / "main.py").write_text("print('hello world')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Readme content", encoding="utf-8")
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "util.py").write_text("def add(a, b): return a + b", encoding="utf-8")

    files = manager.scan_files()
    rel_paths = {f.relative_path for f in files}
    assert "main.py" in rel_paths
    assert "README.md" in rel_paths
    assert "src/util.py" in rel_paths

    stats = manager.get_stats()
    assert stats.total_files == 3
    assert stats.extension_counts.get(".py") == 2
    assert stats.extension_counts.get(".md") == 1
    assert stats.total_bytes > 0
    assert stats.total_estimated_tokens > 0


def test_safe_file_reading_and_traversal_prevention(tmp_path: Path):
    manager = WorkspaceManager(tmp_path)
    manager.initialize()

    test_file = tmp_path / "sample.txt"
    test_file.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    content = manager.read_file_safely("sample.txt")
    assert "line 1" in content
    assert "line 2" in content

    # Directory traversal attempt should raise ValueError
    with pytest.raises(ValueError):
        manager.read_file_safely("../../outside.txt")
