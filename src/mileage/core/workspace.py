"""Workspace and file management for m.AI.leage."""

import fnmatch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from mileage.core.config import MileageConfig
from mileage.core.exceptions import (
    WorkspaceAlreadyInitializedError,
    WorkspaceNotInitializedError,
)
from mileage.core.logger import setup_logger


class FileMetadata(BaseModel):
    """Metadata for an individual workspace file."""

    path: str
    relative_path: str
    size_bytes: int
    extension: str
    estimated_tokens: int
    modified_at: str


class WorkspaceStats(BaseModel):
    """Aggregated workspace statistics."""

    project_name: str
    workspace_root: str
    total_files: int = 0
    total_bytes: int = 0
    total_estimated_tokens: int = 0
    extension_counts: Dict[str, int] = Field(default_factory=dict)
    scanned_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkspaceManifest(BaseModel):
    """Manifest file stored in .mileage/manifest.json."""

    project_name: str
    created_at: str
    version: str
    last_scanned: Optional[str] = None
    file_count: int = 0


class WorkspaceManager:
    """Manages workspace lifecycle, file discovery, and local configuration."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = (root_dir or Path.cwd()).resolve()
        self.mileage_dir = self.root_dir / ".mileage"
        self.config_path = self.mileage_dir / "config.json"
        self.manifest_path = self.mileage_dir / "manifest.json"
        self.logs_dir = self.mileage_dir / "logs"
        self.metrics_path = self.mileage_dir / "metrics.jsonl"
        self._config: Optional[MileageConfig] = None
        self.logger = setup_logger(
            "mileage.workspace",
            log_file=self.logs_dir / "mileage.log" if self.is_initialized() else None,
        )

    @classmethod
    def find_workspace(cls, start_dir: Optional[Path] = None) -> Optional["WorkspaceManager"]:
        """Traverse upwards to find the nearest .mileage workspace directory."""
        current = (start_dir or Path.cwd()).resolve()
        for directory in [current, *current.parents]:
            if (directory / ".mileage").is_dir():
                return cls(directory)
        return None

    def is_initialized(self) -> bool:
        """Check if current workspace has been initialized with .mileage."""
        return self.mileage_dir.is_dir() and self.config_path.is_file()

    def get_config(self) -> MileageConfig:
        """Get or load workspace configuration."""
        if self._config is None:
            self._config = MileageConfig.load_from_dir(self.root_dir)
        return self._config

    def initialize(
        self,
        project_name: Optional[str] = None,
        force: bool = False,
        preferred_model: Optional[str] = None,
    ) -> MileageConfig:
        """Initialize a new m.AI.leage workspace."""
        if self.is_initialized() and not force:
            raise WorkspaceAlreadyInitializedError(
                f"Workspace already initialized at {self.root_dir}"
            )

        name = project_name or self.root_dir.name
        self.mileage_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create or ensure metrics file exists
        if not self.metrics_path.exists():
            self.metrics_path.touch()

        # Build initial config
        config = MileageConfig()
        config.workspace.project_name = name
        if preferred_model:
            config.ollama.default_model = preferred_model

        config.save_to_dir(self.root_dir)
        self._config = config

        # Write workspace manifest
        manifest = WorkspaceManifest(
            project_name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            version=config.version,
        )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        # Write local .gitignore inside .mileage
        gitignore_path = self.mileage_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write("# Ignore logs and temp files\nlogs/\n*.tmp\n")

        self.logger.info("Workspace initialized successfully at %s", self.root_dir)
        return config

    def _should_ignore(self, path: Path, ignore_patterns: List[str]) -> bool:
        """Determine whether a path matches any ignore patterns."""
        rel_str = str(path.relative_to(self.root_dir)).replace("\\", "/")
        parts = rel_str.split("/")

        for pattern in ignore_patterns:
            pattern_clean = pattern.strip("/")
            if fnmatch.fnmatch(path.name, pattern_clean):
                return True
            if any(fnmatch.fnmatch(part, pattern_clean) for part in parts):
                return True
            if fnmatch.fnmatch(rel_str, pattern_clean):
                return True
        return False

    def scan_files(self) -> List[FileMetadata]:
        """Scan workspace files respecting ignore patterns and limits."""
        config = self.get_config()
        ignore_patterns = config.workspace.ignore_patterns
        max_size = config.workspace.max_file_size_bytes
        max_depth = config.workspace.max_scan_depth

        files: List[FileMetadata] = []

        def recurse(directory: Path, depth: int):
            if depth > max_depth:
                return

            try:
                entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
            except PermissionError:
                return

            for entry in entries:
                if self._should_ignore(entry, ignore_patterns):
                    continue

                if entry.is_dir():
                    recurse(entry, depth + 1)
                elif entry.is_file():
                    try:
                        stat = entry.stat()
                        size = stat.st_size
                        # Skip oversized files
                        if size > max_size:
                            continue

                        # Estimate tokens (~4 characters per token for source files)
                        estimated_tokens = max(1, size // 4)
                        rel_path = str(entry.relative_to(self.root_dir)).replace("\\", "/")
                        modified = datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat()

                        files.append(
                            FileMetadata(
                                path=str(entry),
                                relative_path=rel_path,
                                size_bytes=size,
                                extension=entry.suffix.lower() or "no_ext",
                                estimated_tokens=estimated_tokens,
                                modified_at=modified,
                            )
                        )
                    except (PermissionError, OSError):
                        continue

        recurse(self.root_dir, 0)
        return files

    def get_stats(self) -> WorkspaceStats:
        """Compute aggregated statistics for the workspace."""
        config = self.get_config()
        files = self.scan_files()

        extension_counts: Dict[str, int] = {}
        total_bytes = 0
        total_tokens = 0

        for f in files:
            extension_counts[f.extension] = extension_counts.get(f.extension, 0) + 1
            total_bytes += f.size_bytes
            total_tokens += f.estimated_tokens

        # Update manifest last_scanned
        if self.manifest_path.exists():
            try:
                manifest_data = WorkspaceManifest.model_validate_json(
                    self.manifest_path.read_text(encoding="utf-8")
                )
                manifest_data.last_scanned = datetime.now(timezone.utc).isoformat()
                manifest_data.file_count = len(files)
                self.manifest_path.write_text(
                    manifest_data.model_dump_json(indent=2), encoding="utf-8"
                )
            except Exception:
                pass

        return WorkspaceStats(
            project_name=config.workspace.project_name,
            workspace_root=str(self.root_dir),
            total_files=len(files),
            total_bytes=total_bytes,
            total_estimated_tokens=total_tokens,
            extension_counts=extension_counts,
        )

    def read_file_safely(self, relative_path: str, max_lines: int = 500) -> str:
        """Safely read a file from the workspace, preventing directory traversal."""
        target_path = (self.root_dir / relative_path).resolve()
        if not str(target_path).startswith(str(self.root_dir)):
            raise ValueError(f"Path traversal detected: {relative_path}")

        if not target_path.is_file():
            raise FileNotFoundError(f"File not found: {relative_path}")

        lines: List[str] = []
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... [Truncated: reached limit of {max_lines} lines]")
                    break
                lines.append(line)
        return "".join(lines)
