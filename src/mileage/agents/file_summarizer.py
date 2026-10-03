"""Intelligent file summarization with context-size limits.

Provides smart truncation and summarization of workspace files
before they are injected into the planner prompt, ensuring
we stay within Gemma's context window without losing key information.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mileage.core.logger import logger


# Rough token estimate: ~4 chars per token for code, ~5 for prose
CHARS_PER_TOKEN = 4

# File-type priorities for context injection (higher = more important)
EXTENSION_PRIORITY: Dict[str, int] = {
    ".py": 10,
    ".ts": 9,
    ".tsx": 9,
    ".js": 9,
    ".jsx": 9,
    ".rs": 8,
    ".go": 8,
    ".java": 7,
    ".toml": 7,
    ".yaml": 7,
    ".yml": 7,
    ".json": 6,
    ".md": 5,
    ".txt": 4,
    ".cfg": 4,
    ".ini": 4,
    ".html": 3,
    ".css": 3,
    ".sql": 5,
    ".sh": 5,
    ".bat": 4,
    ".env": 3,
}

# Lines to keep from top/bottom when truncating
SUMMARY_HEAD_LINES = 30
SUMMARY_TAIL_LINES = 15


def estimate_tokens(text: str) -> int:
    """Estimate token count from raw text."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def get_file_priority(filepath: str) -> int:
    """Return priority score for a file based on its extension."""
    ext = Path(filepath).suffix.lower()
    return EXTENSION_PRIORITY.get(ext, 1)


def smart_truncate(
    content: str,
    max_tokens: int,
    head_lines: int = SUMMARY_HEAD_LINES,
    tail_lines: int = SUMMARY_TAIL_LINES,
) -> Tuple[str, bool]:
    """Truncate file content intelligently, preserving head + tail.

    Returns:
        (truncated_content, was_truncated)
    """
    if estimate_tokens(content) <= max_tokens:
        return content, False

    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines <= head_lines + tail_lines + 2:
        # Small file but heavy content — just hard-cut
        max_chars = max_tokens * CHARS_PER_TOKEN
        return content[:max_chars] + "\n... [truncated]", True

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    omitted = total_lines - head_lines - tail_lines

    truncated = (
        "".join(head)
        + f"\n... [{omitted} lines omitted — {estimate_tokens(''.join(lines[head_lines:-tail_lines]))} tokens saved] ...\n"
        + "".join(tail)
    )

    # Final safety check
    if estimate_tokens(truncated) > max_tokens:
        max_chars = max_tokens * CHARS_PER_TOKEN
        return truncated[:max_chars] + "\n... [hard truncated]", True

    return truncated, True


def summarize_file(
    filepath: str,
    workspace_root: Path,
    max_tokens_per_file: int = 800,
) -> Optional[str]:
    """Read and summarize a single file for planner context.

    Returns a formatted string with file header + truncated content,
    or None if the file cannot be read.
    """
    abs_path = Path(filepath)
    if not abs_path.is_absolute():
        abs_path = (workspace_root / filepath).resolve()

    if not abs_path.is_file():
        logger.debug("File not found for summarization: %s", filepath)
        return None

    try:
        raw = abs_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug("Could not read file %s: %s", filepath, e)
        return None

    try:
        rel_path = abs_path.relative_to(workspace_root)
    except ValueError:
        rel_path = abs_path.name

    content, was_truncated = smart_truncate(raw, max_tokens_per_file)
    total_lines = len(raw.splitlines())
    total_tokens = estimate_tokens(raw)

    header = (
        f"--- FILE: {rel_path} "
        f"({total_lines} lines, ~{total_tokens} tokens"
        f"{', truncated' if was_truncated else ''}) ---"
    )

    return f"{header}\n{content}\n"


def build_file_context(
    file_paths: List[str],
    workspace_root: Path,
    max_total_tokens: int = 4096,
    max_per_file_tokens: int = 800,
) -> Tuple[str, int]:
    """Build a combined file context string within a total token budget.

    Files are sorted by priority, then packed greedily until the budget
    is exhausted. Returns (context_string, tokens_used).
    """
    # Sort by extension priority (descending), then alphabetically
    sorted_paths = sorted(
        file_paths,
        key=lambda p: (-get_file_priority(p), p),
    )

    parts: List[str] = []
    tokens_used = 0

    for fpath in sorted_paths:
        remaining = max_total_tokens - tokens_used
        if remaining < 50:
            parts.append(
                f"\n... [{len(sorted_paths) - len(parts)} more files omitted — token budget exhausted]\n"
            )
            break

        per_file_limit = min(max_per_file_tokens, remaining)
        summary = summarize_file(fpath, workspace_root, per_file_limit)
        if summary is None:
            continue

        summary_tokens = estimate_tokens(summary)
        if tokens_used + summary_tokens > max_total_tokens:
            # Re-truncate to fit
            available_chars = (max_total_tokens - tokens_used) * CHARS_PER_TOKEN
            summary = summary[:available_chars] + "\n... [budget truncated]\n"
            summary_tokens = estimate_tokens(summary)

        parts.append(summary)
        tokens_used += summary_tokens

    return "\n".join(parts), tokens_used
