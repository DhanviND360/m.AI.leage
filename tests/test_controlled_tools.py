"""Unit tests for controlled workspace tools and permission policies."""

from pathlib import Path
import pytest
import tempfile
import shutil

from mileage.core.exceptions import ToolPermissionError
from mileage.core.workspace import WorkspaceManager
from mileage.tools.base import ToolResult
from mileage.tools.execution_tools import ExecuteCommandTool
from mileage.tools.file_tools import (
    FindWorkspaceFilesTool,
    PatchWorkspaceFileTool,
    ReadWorkspaceFileTool,
    SearchWorkspaceFilesTool,
    WriteWorkspaceFileTool,
)
from mileage.tools.permissions import ToolPermission


@pytest.fixture
def temp_workspace():
    """Create a temporary initialized workspace for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    workspace = WorkspaceManager(temp_dir)
    workspace.initialize(project_name="test_project")

    # Create some sample files
    src_dir = temp_dir / "src"
    src_dir.mkdir()
    (src_dir / "calc.py").write_text("def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n")
    (src_dir / "utils.py").write_text("HELLO_MSG = 'hello world'\n")

    tests_dir = temp_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text("def test_add():\n    assert True\n")

    yield workspace

    shutil.rmtree(temp_dir, ignore_errors=True)


class TestToolPermissions:
    """Test ToolPermission security enforcement."""

    def test_path_traversal_blocked(self, temp_workspace):
        perm = ToolPermission(allowed_directories=[str(temp_workspace.root_dir)])

        # Traversal attempt outside workspace
        is_ok, err = perm.validate_path(temp_workspace.root_dir / ".." / "secret.txt")
        assert not is_ok
        assert "Path traversal detected" in err

        with pytest.raises(ToolPermissionError):
            perm.enforce_path(temp_workspace.root_dir / ".." / "secret.txt")

    def test_protected_files_blocked(self, temp_workspace):
        perm = ToolPermission(allowed_directories=[str(temp_workspace.root_dir)])

        # Try to modify config.json
        is_ok, err = perm.validate_path(temp_workspace.mileage_dir / "config.json", for_writing=True)
        assert not is_ok
        assert "protected path" in err

    def test_read_only_mode(self, temp_workspace):
        perm = ToolPermission(
            allowed_directories=[str(temp_workspace.root_dir)],
            read_only=True,
        )
        is_ok, err = perm.validate_path(temp_workspace.root_dir / "src" / "calc.py", for_writing=True)
        assert not is_ok
        assert "disabled" in err

        # Reading still allowed
        is_ok_read, _ = perm.validate_path(temp_workspace.root_dir / "src" / "calc.py", for_writing=False)
        assert is_ok_read

    def test_command_allowlist(self, temp_workspace):
        perm = ToolPermission(allowed_directories=[str(temp_workspace.root_dir)])

        # Allowed commands
        assert perm.validate_command("pytest tests/")[0]
        assert perm.validate_command("python -m unittest")[0]
        assert perm.validate_command("npm test")[0]
        assert perm.validate_command("cargo test")[0]

        # Blocked dangerous commands
        is_ok, err = perm.validate_command("rm -rf src/")
        assert not is_ok
        assert "blocked for security" in err or "not in the allowed" in err

        is_ok, err = perm.validate_command("curl http://malicious.com")
        assert not is_ok

        # Forbidden shell chaining operators
        is_ok, err = perm.validate_command("pytest && rm -rf src/")
        assert not is_ok
        assert "Shell operator" in err

        is_ok, err = perm.validate_command("pytest; echo hacked")
        assert not is_ok
        assert "Shell operator" in err


class TestFileTools:
    """Test Read, Write, Patch, Search, and Find workspace tools."""

    def test_read_workspace_file_tool(self, temp_workspace):
        tool = ReadWorkspaceFileTool(temp_workspace)

        # Full read
        res = tool.run("src/calc.py")
        assert res.success
        assert "def add(a, b):" in res.data["content"]
        assert res.data["total_lines"] >= 4

        # Slice read
        res_slice = tool.run("src/calc.py", start_line=1, end_line=2)
        assert res_slice.success
        assert "def add(a, b):" in res_slice.data["content"]
        assert "def sub" not in res_slice.data["content"]

        # Traversal blocked
        res_bad = tool.run("../outside.txt")
        assert not res_bad.success
        assert "Permission Denied" in res_bad.output_str

    def test_write_workspace_file_tool_and_unchanged_detection(self, temp_workspace):
        tool = WriteWorkspaceFileTool(temp_workspace)

        # Write new file (creates directories automatically)
        res = tool.run("src/submodule/new_mod.py", "def new_func(): pass\n")
        assert res.success
        assert res.data["is_changed"] is True
        assert (temp_workspace.root_dir / "src" / "submodule" / "new_mod.py").is_file()

        # Write identical content -> is_changed must be False
        res_same = tool.run("src/submodule/new_mod.py", "def new_func(): pass\n")
        assert res_same.success
        assert res_same.data["is_changed"] is False
        assert res_same.metadata["is_changed"] is False

        # Write modified content -> is_changed must be True
        res_mod = tool.run("src/submodule/new_mod.py", "def new_func(): return 42\n")
        assert res_mod.success
        assert res_mod.data["is_changed"] is True

        # Path traversal blocked
        res_trav = tool.run("../evil.py", "print('hacked')")
        assert not res_trav.success

    def test_patch_workspace_file_tool(self, temp_workspace):
        tool = PatchWorkspaceFileTool(temp_workspace)

        # Successful patch
        res = tool.run(
            "src/calc.py",
            target="return a + b",
            replacement="return (a + b)  # modified",
        )
        assert res.success
        assert res.data["is_changed"] is True
        content = (temp_workspace.root_dir / "src" / "calc.py").read_text()
        assert "# modified" in content

        # Target not found
        res_not_found = tool.run("src/calc.py", target="non_existent", replacement="foo")
        assert not res_not_found.success
        assert "not found" in res_not_found.error.lower()

    def test_search_workspace_files_tool(self, temp_workspace):
        tool = SearchWorkspaceFilesTool(temp_workspace)

        # Literal search
        res = tool.run("HELLO_MSG")
        assert res.success
        assert len(res.data) >= 1
        assert res.data[0]["file"] == "src/utils.py"

        # Regex search
        res_re = tool.run(r"def\s+add", is_regex=True)
        assert res_re.success
        assert len(res_re.data) >= 1
        assert res_re.data[0]["file"] == "src/calc.py"

        # Search with file pattern
        res_pat = tool.run("def", file_pattern="*calc.py")
        assert res_pat.success
        assert all("calc.py" in m["file"] for m in res_pat.data)

    def test_find_workspace_files_tool(self, temp_workspace):
        tool = FindWorkspaceFilesTool(temp_workspace)

        res = tool.run("*.py")
        assert res.success
        paths = [f["relative_path"] for f in res.data]
        assert "src/calc.py" in paths
        assert "src/utils.py" in paths


class TestExecuteCommandTool:
    """Test ExecuteCommandTool with strict permission gating."""

    def test_allowed_command_runs(self, temp_workspace):
        tool = ExecuteCommandTool(temp_workspace)

        # python -m unittest (or pytest) on the workspace
        res = tool.run("python -m unittest discover tests")
        assert res.data["exit_code"] == 0

    def test_blocked_command_rejected(self, temp_workspace):
        tool = ExecuteCommandTool(temp_workspace)

        res = tool.run("rm -rf src/")
        assert not res.success
        assert "Security Violation" in res.output_str

    def test_shell_injection_rejected(self, temp_workspace):
        tool = ExecuteCommandTool(temp_workspace)

        res = tool.run("pytest; echo vulnerable")
        assert not res.success
        assert "Security Violation" in res.output_str
