"""Tests for local agent tools."""

from pathlib import Path
from mileage.core.workspace import WorkspaceManager
from mileage.tools.file_tools import (
    ListWorkspaceFilesTool,
    ReadWorkspaceFileTool,
    WorkspaceOverviewTool,
)


def test_file_tools(tmp_path: Path):
    manager = WorkspaceManager(tmp_path)
    manager.initialize(project_name="tool-test")

    (tmp_path / "hello.py").write_text("print('hello')", encoding="utf-8")

    # Read Tool
    read_tool = ReadWorkspaceFileTool(manager)
    res = read_tool.run(relative_path="hello.py")
    assert res.success
    assert "print('hello')" in res.output_str

    # List Tool
    list_tool = ListWorkspaceFilesTool(manager)
    list_res = list_tool.run()
    assert list_res.success
    assert "hello.py" in list_res.output_str

    # Overview Tool
    overview_tool = WorkspaceOverviewTool(manager)
    overview_res = overview_tool.run()
    assert overview_res.success
    assert "tool-test" in overview_res.output_str
