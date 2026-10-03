"""Controlled execution tools with strict permissions."""

from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from mileage.core.logger import logger
from mileage.core.workspace import WorkspaceManager
from mileage.tools.base import BaseTool, ToolResult
from mileage.tools.permissions import ToolPermission


class ExecuteCommandTool(BaseTool):
    """Safely executes an approved project command with strict permission verification."""

    name = "execute_command"
    description = (
        "Execute an approved project command (e.g. 'pytest', 'python -m unittest', 'npm test') "
        "within the workspace root. Never permits unrestricted or destructive shell execution."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The approved command string to execute (e.g. 'pytest tests/')",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Timeout in seconds (default: 60)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace: WorkspaceManager, permission: Optional[ToolPermission] = None):
        self.workspace = workspace
        self.permission = permission or ToolPermission(
            allowed_directories=[str(workspace.root_dir)]
        )

    def run(self, command: str, timeout_seconds: Optional[float] = None) -> ToolResult:
        start_time = time.perf_counter()

        # 1. Strict permission check
        is_ok, err = self.permission.validate_command(command)
        if not is_ok:
            return ToolResult(
                success=False,
                error=err,
                output_str=f"Security Violation: {err}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
                metadata={"command": command, "permission_denied": True},
            )

        timeout = timeout_seconds or self.permission.max_execution_timeout_seconds

        # 2. Parse command arguments safely
        try:
            # On Windows, keep standard quotes if possible
            args = shlex.split(command, posix=(sys.platform != "win32"))
        except Exception:
            args = command.split()

        # Normalize python path if command begins with python
        if args and args[0] in ["python", "python3", "py"]:
            args[0] = sys.executable

        logger.info("Executing controlled command: %s (cwd=%s)", args, self.workspace.root_dir)

        try:
            proc = subprocess.run(
                args,
                cwd=str(self.workspace.root_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            stdout_clean = proc.stdout.strip()
            stderr_clean = proc.stderr.strip()

            combined = []
            if stdout_clean:
                combined.append(stdout_clean)
            if stderr_clean:
                combined.append(f"[stderr]\n{stderr_clean}")
            output_str = "\n".join(combined) if combined else "[Command produced no output]"

            return ToolResult(
                success=(proc.returncode == 0),
                data={
                    "command": command,
                    "exit_code": proc.returncode,
                    "stdout": stdout_clean,
                    "stderr": stderr_clean,
                    "duration_ms": round(duration_ms, 2),
                },
                output_str=output_str,
                duration_ms=round(duration_ms, 2),
                metadata={
                    "exit_code": proc.returncode,
                    "command": command,
                },
            )

        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"Command timed out after {timeout} seconds: '{command}'"
            return ToolResult(
                success=False,
                error=err_msg,
                output_str=err_msg,
                duration_ms=round(duration_ms, 2),
                metadata={"timed_out": True},
            )
        except FileNotFoundError as fnf:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"Executable not found on system: {args[0] if args else command}"
            return ToolResult(
                success=False,
                error=err_msg,
                output_str=err_msg,
                duration_ms=round(duration_ms, 2),
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Command execution error: {str(e)}",
                duration_ms=round(duration_ms, 2),
            )
