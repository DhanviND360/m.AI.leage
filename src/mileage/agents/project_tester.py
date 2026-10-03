"""Automatic project-native test discovery, execution, and output parsing."""

import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from mileage.core.logger import logger
from mileage.tools.permissions import ToolPermission


class TestRunResult(BaseModel):
    """Structured result of executing project-native tests."""

    __test__ = False

    passed: bool
    exit_code: int
    command: str
    framework: str = "generic"
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errors: int = 0
    total_tests: int = 0
    duration_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    failure_summary: str = ""
    raw_output: str = ""


class ProjectTester:
    """Discovers and automatically executes project-native tests under strict permissions."""

    def __init__(
        self,
        workspace_root: Path,
        permission: Optional[ToolPermission] = None,
        custom_test_cmd: Optional[str] = None,
    ):
        self.workspace_root = workspace_root.resolve()
        self.permission = permission or ToolPermission(
            allowed_directories=[str(self.workspace_root)]
        )
        self.custom_test_cmd = custom_test_cmd

    def detect_framework(self) -> Tuple[str, str]:
        """Detect the project test framework and return (framework_name, default_command)."""
        if self.custom_test_cmd:
            return "custom", self.custom_test_cmd

        # 1. Python: pytest or unittest
        pyproject = self.workspace_root / "pyproject.toml"
        pytest_ini = self.workspace_root / "pytest.ini"
        setup_cfg = self.workspace_root / "setup.cfg"
        tests_dir = self.workspace_root / "tests"

        has_python_tests = False
        if tests_dir.is_dir():
            has_python_tests = any(tests_dir.glob("test_*.py")) or any(tests_dir.glob("*_test.py"))

        if pytest_ini.is_file() or (pyproject.is_file() and "pytest" in pyproject.read_text(encoding="utf-8", errors="ignore")) or has_python_tests:
            return "pytest", "pytest"

        # Check for python files with unittest if no pytest
        if has_python_tests:
            return "unittest", "python -m unittest"

        # 2. Node / JS
        package_json = self.workspace_root / "package.json"
        if package_json.is_file():
            try:
                pkg_data = json.loads(package_json.read_text(encoding="utf-8", errors="ignore"))
                scripts = pkg_data.get("scripts", {})
                if "test" in scripts:
                    return "npm", "npm test"
            except Exception:
                return "npm", "npm test"

        # 3. Rust
        cargo_toml = self.workspace_root / "Cargo.toml"
        if cargo_toml.is_file():
            return "cargo", "cargo test"

        # 4. Go
        go_mod = self.workspace_root / "go.mod"
        if go_mod.is_file():
            return "go", "go test ./..."

        # Default fallback: try pytest if any .py files exist, else empty
        py_files = list(self.workspace_root.glob("*.py")) + list(self.workspace_root.glob("src/**/*.py"))
        if py_files:
            return "pytest", "pytest"

        return "unknown", ""

    def run(self, timeout_seconds: float = 60.0) -> TestRunResult:
        """Run project tests automatically and parse the results."""
        framework, cmd_str = self.detect_framework()

        if not cmd_str:
            return TestRunResult(
                passed=True,
                exit_code=0,
                command="",
                framework="none",
                raw_output="No test framework or test command detected in workspace.",
                failure_summary="",
            )

        # Validate permission
        is_ok, err = self.permission.validate_command(cmd_str)
        if not is_ok:
            return TestRunResult(
                passed=False,
                exit_code=1,
                command=cmd_str,
                framework=framework,
                failure_summary=f"Permission Denied: {err}",
                raw_output=f"Security Policy Blocked Command: {cmd_str}\n{err}",
            )

        # Prepare execution args
        import shlex
        try:
            args = shlex.split(cmd_str, posix=(sys.platform != "win32"))
        except Exception:
            args = cmd_str.split()

        if args and args[0] in ["python", "python3", "py"]:
            args[0] = sys.executable
        elif args and args[0] == "pytest":
            args = [sys.executable, "-m", "pytest"] + args[1:]

        import os
        env = dict(os.environ)
        cur_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.workspace_root}{os.pathsep}{self.workspace_root / 'src'}{os.pathsep}{cur_pythonpath}"

        start_time = time.perf_counter()
        logger.info("ProjectTester running: %s (cwd=%s)", args, self.workspace_root)

        try:
            proc = subprocess.run(
                args,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                env=env,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            raw_output = f"{stdout}\n{stderr}".strip()

            return self._parse_output(
                framework=framework,
                command=cmd_str,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                raw_output=raw_output,
                duration_ms=duration_ms,
            )

        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start_time) * 1000
            msg = f"Test execution timed out after {timeout_seconds}s: '{cmd_str}'"
            return TestRunResult(
                passed=False,
                exit_code=124,
                command=cmd_str,
                framework=framework,
                duration_ms=round(duration_ms, 2),
                failure_summary=msg,
                raw_output=msg,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            msg = f"Test execution error: {str(e)}"
            return TestRunResult(
                passed=False,
                exit_code=1,
                command=cmd_str,
                framework=framework,
                duration_ms=round(duration_ms, 2),
                failure_summary=msg,
                raw_output=msg,
            )

    def run_tests(self, timeout_seconds: float = 60.0) -> TestRunResult:
        """Alias for run() to execute project-native tests."""
        return self.run(timeout_seconds=timeout_seconds)

    def _parse_output(
        self,
        framework: str,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        raw_output: str,
        duration_ms: float,
    ) -> TestRunResult:
        """Parse framework-specific test output for counts and failure summaries."""
        passed = (exit_code == 0)
        tests_passed = 0
        tests_failed = 0
        tests_errors = 0
        failure_summary = ""

        if framework in ["pytest", "unittest", "custom"]:
            # Match pytest patterns: "X passed, Y failed, Z error in 1.23s"
            # e.g., "=== 87 passed in 5.48s ==="
            pass_match = re.search(r"(\d+)\s+passed", raw_output, re.IGNORECASE)
            fail_match = re.search(r"(\d+)\s+failed", raw_output, re.IGNORECASE)
            err_match = re.search(r"(\d+)\s+error", raw_output, re.IGNORECASE)

            if pass_match:
                tests_passed = int(pass_match.group(1))
            if fail_match:
                tests_failed = int(fail_match.group(1))
            if err_match:
                tests_errors = int(err_match.group(1))

            # If not passed, extract FAILURES block or ERRORS block
            if not passed:
                failure_blocks = []
                # Look for "FAILURES", "ERRORS", "Traceback"
                in_failure = False
                lines = raw_output.splitlines()
                extracted_lines = []
                for line in lines:
                    if "=== FAILURES ===" in line or "=== ERRORS ===" in line or line.startswith("FAILED "):
                        in_failure = True
                    if in_failure:
                        extracted_lines.append(line)
                        if len(extracted_lines) >= 40:
                            extracted_lines.append("... [truncated failure output]")
                            break

                if extracted_lines:
                    failure_summary = "\n".join(extracted_lines)
                else:
                    # Fallback to last 20 lines of output
                    failure_summary = "\n".join(lines[-25:])

        elif framework == "npm":
            # Check npm / vitest / jest output
            pass_match = re.search(r"Tests?:\s*(\d+)\s*passed", raw_output, re.IGNORECASE)
            fail_match = re.search(r"Tests?:\s*(\d+)\s*failed", raw_output, re.IGNORECASE)
            if pass_match:
                tests_passed = int(pass_match.group(1))
            if fail_match:
                tests_failed = int(fail_match.group(1))
            if not passed:
                lines = raw_output.splitlines()
                failure_summary = "\n".join(lines[-30:])

        elif framework == "cargo":
            pass_match = re.search(r"test result: [^;]+; (\d+) passed; (\d+) failed", raw_output)
            if pass_match:
                tests_passed = int(pass_match.group(1))
                tests_failed = int(pass_match.group(2))
            if not passed:
                lines = raw_output.splitlines()
                failure_summary = "\n".join(lines[-30:])

        if passed and not tests_passed:
            tests_passed = 1  # At least 1 passed if exit code was 0

        total_tests = tests_passed + tests_failed + tests_errors

        return TestRunResult(
            passed=passed,
            exit_code=exit_code,
            command=command,
            framework=framework,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_errors=tests_errors,
            total_tests=total_tests,
            duration_ms=round(duration_ms, 2),
            stdout=stdout,
            stderr=stderr,
            failure_summary=failure_summary.strip(),
            raw_output=raw_output,
        )
