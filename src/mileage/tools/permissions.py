"""Security and permission policies for agent tool execution."""

from pathlib import Path
import re
import shlex
import sys
from typing import List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from mileage.core.exceptions import ToolPermissionError


DEFAULT_ALLOWED_COMMANDS = {
    # Python testing and linting
    "pytest",
    "python -m pytest",
    "python -m unittest",
    "py.test",
    "ruff",
    "mypy",
    "flake8",
    "python -m build",
    # Node / JS testing and building
    "npm test",
    "npm run test",
    "npm run build",
    "npx vitest",
    "npx vitest run",
    "npx jest",
    "yarn test",
    "pnpm test",
    # Rust testing
    "cargo test",
    "cargo check",
    "cargo build",
    # Go testing
    "go test",
    "go test ./...",
    "go build",
}

DEFAULT_BLOCKED_COMMANDS = {
    "rm",
    "del",
    "erase",
    "rmdir",
    "rd",
    "curl",
    "wget",
    "bash",
    "sh",
    "zsh",
    "powershell",
    "cmd",
    "cmd.exe",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "sudo",
    "su",
    "chmod",
    "chown",
    "kill",
    "taskkill",
    "shutdown",
    "format",
    "mkfs",
    "dd",
}

# Dangerous shell operators that could chain or redirect arbitrary commands
FORBIDDEN_OPERATORS = [";", "&&", "||", "|", ">", "<", ">>", "`", "$(", "${"]


class ToolPermission(BaseModel):
    """Enforces explicit, controlled permissions for agent tool execution."""

    allow_read: bool = True
    allow_write: bool = True
    allow_search: bool = True
    allow_execute: bool = True
    read_only: bool = False

    allowed_commands: Set[str] = Field(default_factory=lambda: set(DEFAULT_ALLOWED_COMMANDS))
    blocked_commands: Set[str] = Field(default_factory=lambda: set(DEFAULT_BLOCKED_COMMANDS))

    allowed_directories: List[str] = Field(default_factory=list)
    blocked_patterns: List[str] = Field(
        default_factory=lambda: [".git", ".mileage/config.json", ".mileage/manifest.json"]
    )
    max_execution_timeout_seconds: float = 60.0

    def validate_path(self, path: Path | str, for_writing: bool = False) -> Tuple[bool, Optional[str]]:
        """Validate if a path is permitted for reading or writing."""
        if for_writing and (self.read_only or not self.allow_write):
            return False, "Permission denied: File write operations are disabled."

        if not for_writing and not self.allow_read:
            return False, "Permission denied: File read operations are disabled."

        target_path = Path(path).resolve()
        target_str = str(target_path).replace("\\", "/")

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in target_str:
                return False, f"Permission denied: Access to protected path pattern '{pattern}' is blocked."

        # Check if target is inside any allowed directory
        if self.allowed_directories:
            inside_allowed = False
            for allowed in self.allowed_directories:
                allowed_res = Path(allowed).resolve()
                try:
                    target_path.relative_to(allowed_res)
                    inside_allowed = True
                    break
                except ValueError:
                    continue

            if not inside_allowed:
                return (
                    False,
                    f"Permission denied: Path traversal detected. Path '{path}' is outside allowed directories: {self.allowed_directories}",
                )

        return True, None

    def enforce_path(self, path: Path | str, for_writing: bool = False) -> Path:
        """Validate path and raise ToolPermissionError if not permitted."""
        is_ok, err = self.validate_path(path, for_writing=for_writing)
        if not is_ok:
            raise ToolPermissionError(err or "Path permission denied")
        return Path(path).resolve()

    def validate_command(self, command: str | List[str]) -> Tuple[bool, Optional[str]]:
        """Validate if a command is permitted for execution."""
        if not self.allow_execute:
            return False, "Permission denied: Command execution is disabled."

        cmd_str = command if isinstance(command, str) else " ".join(command)
        cmd_clean = cmd_str.strip()

        if not cmd_clean:
            return False, "Empty command."

        # Check forbidden shell operators
        for op in FORBIDDEN_OPERATORS:
            if op in cmd_clean:
                return (
                    False,
                    f"Permission denied: Shell operator '{op}' is forbidden to prevent shell injection.",
                )

        # Parse command tokens safely
        try:
            tokens = shlex.split(cmd_clean, posix=(sys.platform != "win32"))
        except Exception:
            tokens = cmd_clean.split()

        if not tokens:
            return False, "Empty command after tokenizing."

        base_binary = Path(tokens[0]).name.lower()
        if base_binary.endswith(".exe"):
            base_binary = base_binary[:-4]

        # Check explicit blocked commands
        if base_binary in self.blocked_commands or tokens[0].lower() in self.blocked_commands:
            return False, f"Permission denied: Command '{tokens[0]}' is explicitly blocked for security."

        # Normalize python invocation (e.g. sys.executable or python3 -> python)
        normalized_tokens = list(tokens)
        if normalized_tokens[0] in [sys.executable, "python", "python3", "py"]:
            normalized_tokens[0] = "python"

        normalized_cmd = " ".join(normalized_tokens)

        # Match against allowed commands (exact prefix match)
        is_allowed = False
        for allowed in self.allowed_commands:
            allowed_tokens = allowed.split()
            # If command matches or starts with allowed command tokens
            if len(normalized_tokens) >= len(allowed_tokens):
                if normalized_tokens[: len(allowed_tokens)] == allowed_tokens:
                    is_allowed = True
                    break
            # Also allow direct binary match if allowed is a single binary name
            if len(allowed_tokens) == 1 and normalized_tokens[0] == allowed_tokens[0]:
                is_allowed = True
                break

        if not is_allowed:
            return (
                False,
                f"Permission denied: Command '{cmd_str}' is not in the allowed commands policy. "
                f"Allowed prefixes: {sorted(list(self.allowed_commands))}",
            )

        return True, None

    def enforce_command(self, command: str | List[str]) -> str:
        """Validate command and raise ToolPermissionError if not permitted."""
        is_ok, err = self.validate_command(command)
        if not is_ok:
            raise ToolPermissionError(err or "Command permission denied")
        return command if isinstance(command, str) else " ".join(command)
