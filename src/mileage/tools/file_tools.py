"""Local workspace file tools with strict permission controls."""

import difflib
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from mileage.core.workspace import WorkspaceManager
from mileage.tools.base import BaseTool, ToolResult
from mileage.tools.permissions import ToolPermission


class ReadWorkspaceFileTool(BaseTool):
    """Tool to safely read a file from the workspace with line range support."""

    name = "read_file"
    description = (
        "Read contents of a workspace file by relative path. "
        "Supports optional start_line and end_line parameters."
    )
    parameters = {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string", "description": "Workspace-relative file path"},
            "start_line": {"type": "integer", "description": "Starting line number (1-indexed)"},
            "end_line": {"type": "integer", "description": "Ending line number (inclusive)"},
            "max_lines": {"type": "integer", "description": "Maximum lines to read", "default": 500},
        },
        "required": ["relative_path"],
    }

    def __init__(self, workspace: WorkspaceManager, permission: Optional[ToolPermission] = None):
        self.workspace = workspace
        self.permission = permission or ToolPermission(
            allowed_directories=[str(workspace.root_dir)]
        )

    def run(
        self,
        relative_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_lines: int = 500,
    ) -> ToolResult:
        start_time = time.perf_counter()
        target_path = (self.workspace.root_dir / relative_path).resolve()

        # Permission and traversal check
        is_ok, err = self.permission.validate_path(target_path, for_writing=False)
        if not is_ok:
            return ToolResult(
                success=False,
                error=err,
                output_str=f"Permission Denied: {err}",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        if not target_path.is_file():
            return ToolResult(
                success=False,
                error=f"File not found: {relative_path}",
                output_str=f"File not found: {relative_path}",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            s_idx = max(0, (start_line - 1)) if start_line is not None else 0
            e_idx = min(total_lines, end_line) if end_line is not None else total_lines

            # Limit slice to max_lines
            if (e_idx - s_idx) > max_lines:
                e_idx = s_idx + max_lines
                truncated = True
            else:
                truncated = False

            selected_lines = all_lines[s_idx:e_idx]
            content = "".join(selected_lines)
            if truncated:
                content += f"\n... [Truncated: displayed {max_lines} lines of {total_lines}]"

            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=True,
                data={
                    "path": relative_path,
                    "content": content,
                    "total_lines": total_lines,
                    "start_line": s_idx + 1,
                    "end_line": e_idx,
                },
                output_str=content,
                duration_ms=round(duration_ms, 2),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Error reading file '{relative_path}': {str(e)}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )


class WriteWorkspaceFileTool(BaseTool):
    """Tool to safely write or create a file in the workspace, detecting unchanged edits."""

    name = "write_file"
    description = (
        "Write full content to a file in the workspace. "
        "Creates parent directories if necessary and tracks diffs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string", "description": "Workspace-relative file path"},
            "content": {"type": "string", "description": "Complete file content to write"},
        },
        "required": ["relative_path", "content"],
    }

    def __init__(self, workspace: WorkspaceManager, permission: Optional[ToolPermission] = None):
        self.workspace = workspace
        self.permission = permission or ToolPermission(
            allowed_directories=[str(workspace.root_dir)]
        )

    def run(self, relative_path: str, content: str) -> ToolResult:
        start_time = time.perf_counter()
        target_path = (self.workspace.root_dir / relative_path).resolve()

        # Permission and traversal check
        is_ok, err = self.permission.validate_path(target_path, for_writing=True)
        if not is_ok:
            return ToolResult(
                success=False,
                error=err,
                output_str=f"Permission Denied: {err}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        try:
            # Read existing content if file exists
            old_content = ""
            file_existed = target_path.is_file()
            if file_existed:
                with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()

            is_changed = (old_content != content)

            # Compute diff statistics
            old_lines = old_content.splitlines(keepends=True)
            new_lines = content.splitlines(keepends=True)
            diff = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{relative_path}",
                    tofile=f"b/{relative_path}",
                )
            )
            added_count = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
            removed_count = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

            # Ensure parent directories exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)

            duration_ms = (time.perf_counter() - start_time) * 1000
            diff_str = "".join(diff[:50])

            summary = (
                f"Wrote {len(content)} bytes to '{relative_path}' "
                f"(+{added_count}, -{removed_count} lines). Changed: {is_changed}"
            )

            return ToolResult(
                success=True,
                data={
                    "path": relative_path,
                    "is_changed": is_changed,
                    "existed": file_existed,
                    "lines_added": added_count,
                    "lines_removed": removed_count,
                    "bytes_written": len(content),
                    "diff": diff_str,
                },
                output_str=summary,
                duration_ms=round(duration_ms, 2),
                metadata={"is_changed": is_changed, "path": relative_path},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Error writing file '{relative_path}': {str(e)}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )


class PatchWorkspaceFileTool(BaseTool):
    """Tool to safely patch or replace a substring in an existing file."""

    name = "patch_file"
    description = (
        "Replace an exact target substring with replacement content in an existing file. "
        "Fails if target substring is not found."
    )
    parameters = {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string", "description": "Workspace-relative file path"},
            "target": {"type": "string", "description": "Exact text substring to replace"},
            "replacement": {"type": "string", "description": "New replacement text"},
            "allow_multiple": {
                "type": "boolean",
                "description": "Whether to replace multiple occurrences (default false)",
            },
        },
        "required": ["relative_path", "target", "replacement"],
    }

    def __init__(self, workspace: WorkspaceManager, permission: Optional[ToolPermission] = None):
        self.workspace = workspace
        self.permission = permission or ToolPermission(
            allowed_directories=[str(workspace.root_dir)]
        )

    def run(
        self,
        relative_path: str,
        target: str,
        replacement: str,
        allow_multiple: bool = False,
    ) -> ToolResult:
        start_time = time.perf_counter()
        target_path = (self.workspace.root_dir / relative_path).resolve()

        # Permission and traversal check
        is_ok, err = self.permission.validate_path(target_path, for_writing=True)
        if not is_ok:
            return ToolResult(
                success=False,
                error=err,
                output_str=f"Permission Denied: {err}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        if not target_path.is_file():
            return ToolResult(
                success=False,
                error=f"File not found: {relative_path}",
                output_str=f"File not found: {relative_path}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        try:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            occurrences = content.count(target)
            if occurrences == 0:
                return ToolResult(
                    success=False,
                    error=f"Target string not found in '{relative_path}'",
                    output_str=f"Target text was not found in '{relative_path}'.",
                    duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
                )

            if occurrences > 1 and not allow_multiple:
                return ToolResult(
                    success=False,
                    error=f"Target occurs {occurrences} times in '{relative_path}'. Set allow_multiple=True to replace all.",
                    output_str=f"Target string occurs {occurrences} times. Must be unique or set allow_multiple=True.",
                    duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
                )

            count = -1 if allow_multiple else 1
            new_content = content.replace(target, replacement, count)
            is_changed = (content != new_content)

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            duration_ms = (time.perf_counter() - start_time) * 1000
            summary = (
                f"Successfully patched '{relative_path}' ({occurrences} occurrence(s) replaced). "
                f"Changed: {is_changed}"
            )
            return ToolResult(
                success=True,
                data={
                    "path": relative_path,
                    "is_changed": is_changed,
                    "occurrences_replaced": occurrences if allow_multiple else 1,
                },
                output_str=summary,
                duration_ms=round(duration_ms, 2),
                metadata={"is_changed": is_changed, "path": relative_path},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Error patching '{relative_path}': {str(e)}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )


class SearchWorkspaceFilesTool(BaseTool):
    """Tool to search for text patterns or regex across workspace files."""

    name = "search_files"
    description = (
        "Search for a text query or regular expression across all workspace files. "
        "Returns matching files, line numbers, and line contents."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search string or regular expression"},
            "is_regex": {"type": "boolean", "description": "Treat query as regex (default false)"},
            "file_pattern": {"type": "string", "description": "Glob filter (e.g. *.py, src/*)"},
            "max_results": {"type": "integer", "description": "Max matching lines (default 50)"},
        },
        "required": ["query"],
    }

    def __init__(self, workspace: WorkspaceManager, permission: Optional[ToolPermission] = None):
        self.workspace = workspace
        self.permission = permission or ToolPermission(
            allowed_directories=[str(workspace.root_dir)]
        )

    def run(
        self,
        query: str,
        is_regex: bool = False,
        file_pattern: Optional[str] = None,
        max_results: int = 50,
    ) -> ToolResult:
        start_time = time.perf_counter()
        if not self.permission.allow_search:
            return ToolResult(
                success=False,
                error="Search permission denied",
                output_str="Permission Denied: File search is disabled.",
                duration_ms=0.0,
            )

        try:
            files = self.workspace.scan_files()
            matches: List[Dict[str, Any]] = []

            # Compile pattern
            pattern = re.compile(query, re.IGNORECASE) if is_regex else None
            lower_query = query.lower()

            for f_meta in files:
                if len(matches) >= max_results:
                    break

                rel_path = f_meta.relative_path
                # Check file pattern filter if provided
                if file_pattern:
                    import fnmatch
                    if not fnmatch.fnmatch(rel_path, file_pattern) and not fnmatch.fnmatch(Path(rel_path).name, file_pattern):
                        continue

                file_path = Path(f_meta.path)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for line_no, line in enumerate(f, 1):
                            matched = False
                            if is_regex and pattern:
                                if pattern.search(line):
                                    matched = True
                            elif lower_query in line.lower():
                                matched = True

                            if matched:
                                matches.append({
                                    "file": rel_path,
                                    "line": line_no,
                                    "content": line.rstrip("\r\n"),
                                })
                                if len(matches) >= max_results:
                                    break
                except Exception:
                    continue

            output_lines = [
                f"{m['file']}:{m['line']}: {m['content']}" for m in matches
            ]
            summary = "\n".join(output_lines) if output_lines else f"No matches found for '{query}'."

            return ToolResult(
                success=True,
                data=matches,
                output_str=summary,
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Search failed: {str(e)}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )


class FindWorkspaceFilesTool(BaseTool):
    """Tool to find files matching a glob pattern in the workspace."""

    name = "find_files"
    description = "Find workspace files matching a glob pattern (e.g. '*.py', 'tests/**', 'src/*')."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (default '*')"},
            "max_results": {"type": "integer", "description": "Max file results (default 100)"},
        },
    }

    def __init__(self, workspace: WorkspaceManager, permission: Optional[ToolPermission] = None):
        self.workspace = workspace
        self.permission = permission or ToolPermission(
            allowed_directories=[str(workspace.root_dir)]
        )

    def run(self, pattern: str = "*", max_results: int = 100) -> ToolResult:
        start_time = time.perf_counter()
        try:
            import fnmatch
            files = self.workspace.scan_files()
            matching = []

            for f in files:
                rel = f.relative_path
                base = Path(rel).name
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(base, pattern):
                    matching.append(f)
                    if len(matching) >= max_results:
                        break

            lines = [
                f"{m.relative_path} ({m.size_bytes} bytes, ~{m.estimated_tokens} tokens)"
                for m in matching
            ]
            summary = "\n".join(lines) if lines else f"No files matched pattern '{pattern}'."

            return ToolResult(
                success=True,
                data=[m.model_dump() for m in matching],
                output_str=summary,
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Find files failed: {str(e)}",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )


class ListWorkspaceFilesTool(BaseTool):
    """Tool to inspect files in the local workspace."""

    name = "list_files"
    description = "List all tracked files in the workspace with their size and token estimates."

    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def run(self) -> ToolResult:
        try:
            files = self.workspace.scan_files()
            formatted = [
                f"{f.relative_path} ({f.size_bytes} bytes, ~{f.estimated_tokens} tokens)"
                for f in files
            ]
            return ToolResult(
                success=True,
                data=[f.model_dump() for f in files],
                output_str="\n".join(formatted) if formatted else "No files found in workspace.",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Error listing files: {str(e)}",
            )


class WorkspaceOverviewTool(BaseTool):
    """Tool to generate a statistical overview of the local workspace."""

    name = "workspace_overview"
    description = "Get summary stats on total files, total bytes, estimated tokens, and file types."

    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    def run(self) -> ToolResult:
        try:
            stats = self.workspace.get_stats()
            summary = (
                f"Workspace '{stats.project_name}': {stats.total_files} files, "
                f"{stats.total_bytes:,} bytes, ~{stats.total_estimated_tokens:,} tokens. "
                f"File types: {dict(stats.extension_counts)}"
            )
            return ToolResult(
                success=True,
                data=stats.model_dump(),
                output_str=summary,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                output_str=f"Error getting workspace stats: {str(e)}",
            )
