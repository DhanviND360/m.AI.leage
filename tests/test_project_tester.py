"""Unit tests for ProjectTester auto test discovery and parsing."""

import json
from pathlib import Path
import pytest
import shutil
import tempfile

from mileage.agents.project_tester import ProjectTester
from mileage.tools.permissions import ToolPermission


@pytest.fixture
def temp_workspace_for_tests():
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestProjectTester:
    """Test project-native test discovery, execution, and parsing."""

    def test_detect_python_pytest(self, temp_workspace_for_tests):
        tests_dir = temp_workspace_for_tests / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_sample.py").write_text("def test_ok(): assert True\n")

        tester = ProjectTester(temp_workspace_for_tests)
        framework, cmd = tester.detect_framework()
        assert framework == "pytest"
        assert cmd == "pytest"

    def test_detect_npm(self, temp_workspace_for_tests):
        pkg = temp_workspace_for_tests / "package.json"
        pkg.write_text(json.dumps({"name": "test-pkg", "scripts": {"test": "vitest"}}))

        tester = ProjectTester(temp_workspace_for_tests)
        framework, cmd = tester.detect_framework()
        assert framework == "npm"
        assert cmd == "npm test"

    def test_detect_cargo(self, temp_workspace_for_tests):
        (temp_workspace_for_tests / "Cargo.toml").write_text("[package]\nname = 'test'\n")

        tester = ProjectTester(temp_workspace_for_tests)
        framework, cmd = tester.detect_framework()
        assert framework == "cargo"
        assert cmd == "cargo test"

    def test_detect_go(self, temp_workspace_for_tests):
        (temp_workspace_for_tests / "go.mod").write_text("module example.com/test\n")

        tester = ProjectTester(temp_workspace_for_tests)
        framework, cmd = tester.detect_framework()
        assert framework == "go"
        assert cmd == "go test ./..."

    def test_run_pytest_passing(self, temp_workspace_for_tests):
        tests_dir = temp_workspace_for_tests / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_calc.py").write_text("def test_one(): assert 1 == 1\ndef test_two(): assert 2 == 2\n")

        tester = ProjectTester(temp_workspace_for_tests)
        res = tester.run()
        assert res.passed is True
        assert res.exit_code == 0
        assert res.tests_passed == 2
        assert res.tests_failed == 0

    def test_run_pytest_failing(self, temp_workspace_for_tests):
        tests_dir = temp_workspace_for_tests / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_calc.py").write_text("def test_fail(): assert 1 == 2, 'Value mismatch'\n")

        tester = ProjectTester(temp_workspace_for_tests)
        res = tester.run()
        assert res.passed is False
        assert res.exit_code != 0
        assert res.tests_failed >= 1
        assert "assert 1 == 2" in res.failure_summary or "Value mismatch" in res.failure_summary

    def test_custom_command_permission_blocked(self, temp_workspace_for_tests):
        perm = ToolPermission(allowed_directories=[str(temp_workspace_for_tests)])
        # Block arbitrary command
        tester = ProjectTester(temp_workspace_for_tests, permission=perm, custom_test_cmd="cat /etc/passwd")
        res = tester.run()
        assert res.passed is False
        assert "Permission Denied" in res.failure_summary
