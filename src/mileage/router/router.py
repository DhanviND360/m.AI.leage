"""Model Router — the top-level orchestrator.

Ties together discovery, capability registry, benchmarks, storage,
and deterministic scoring to automatically select the best local model
for any given task.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from mileage.core.logger import logger
from mileage.models.ollama_client import OllamaClient
from mileage.router.benchmark import run_benchmarks
from mileage.router.discovery import discover_models, get_hardware_info
from mileage.router.registry import populate_all
from mileage.router.schemas import (
    Capability,
    HardwareInfo,
    ModelProfile,
    RouterConfig,
    RoutingDecision,
    TaskCategory,
)
from mileage.router.scorer import ModelScorer
from mileage.router.storage import RouterStorage


class ModelRouter:
    """Automatically discovers, evaluates, and selects the best local model.

    Usage:
        router = ModelRouter(ollama_client, workspace_root=Path("."))
        decision = router.route(TaskCategory.CODE_GENERATION)
        print(decision.selected_model, decision.reason)

    Architecture:
        1. Discovery → probe Ollama for installed models
        2. Registry  → assign capability scores per model
        3. Scorer    → deterministic composite scoring
        4. Storage   → persist benchmarks + decisions in SQLite
        5. Config    → all rules editable as JSON
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        workspace_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        self.client = ollama_client
        self.workspace_root = workspace_root or Path.cwd()

        # Config: load from .mileage/router_config.json or defaults
        mileage_dir = self.workspace_root / ".mileage"
        self._config_path = config_path or (mileage_dir / "router_config.json")
        self.config = RouterConfig.load_from_file(self._config_path)

        # Storage: SQLite in .mileage/router.db
        self._db_path = db_path or (mileage_dir / "router.db")
        self.storage = RouterStorage(self._db_path)

        # Scorer
        self.scorer = ModelScorer(config=self.config, storage=self.storage)

        # Cached profiles
        self._profiles: List[ModelProfile] = []
        self._hardware: Optional[HardwareInfo] = None

    # ── Public API ───────────────────────────────────────────────────────

    def discover(self) -> List[ModelProfile]:
        """Discover all installed models, populate capabilities, and cache.

        Call this once at startup or when models change.
        """
        # Ensure Ollama is running
        self.client.start_daemon(timeout_seconds=5.0)

        # Discover installed models
        self._profiles = discover_models(self.client)

        # Populate capability scores from registry
        populate_all(self._profiles)

        # Enrich with benchmark history from SQLite
        for profile in self._profiles:
            stats = self.storage.get_model_stats(profile.name)
            if stats["total"] > 0:
                profile.avg_latency_ms = stats["avg_latency_ms"]
                profile.avg_tokens_per_sec = stats["avg_tps"]
                profile.benchmark_success_rate = stats["success_rate"]
                profile.benchmark_run_count = int(stats["total"])

            # Cache profile in SQLite
            self.storage.save_profile(profile)

        # Detect hardware
        self._hardware = get_hardware_info()

        logger.info(
            "Router discovered %d model(s) with capabilities populated.",
            len(self._profiles),
        )

        return self._profiles

    def route(
        self,
        task_category: TaskCategory,
        required_capabilities: Optional[List[Capability]] = None,
    ) -> RoutingDecision:
        """Select the best model for a task category.

        Uses deterministic scoring — no LLM involvement.
        The decision is fully transparent and persisted.

        Args:
            task_category: What kind of task needs to run.
            required_capabilities: Override required capabilities from config.

        Returns:
            RoutingDecision with selected model, fallback, reason, and scores.
        """
        # Auto-discover if not done yet
        if not self._profiles:
            self.discover()

        if not self._profiles:
            return RoutingDecision(
                selected_model=self.config.fallback_model or "unknown",
                reason="No models discovered. Ensure Ollama is running with installed models.",
                task_category=task_category,
                confidence=0.0,
            )

        # Filter by required capabilities if overridden
        candidates = self._profiles
        if required_capabilities:
            candidates = [
                p for p in self._profiles
                if all(p.has_capability(c) for c in required_capabilities)
            ]
            if not candidates:
                # Fall back to all profiles if filter is too strict
                candidates = self._profiles

        decision = self.scorer.select_best(candidates, task_category)

        logger.info(
            "Routed %s → %s (confidence=%.2f, reason=%s)",
            task_category.value,
            decision.selected_model,
            decision.confidence,
            decision.reason[:100],
        )

        return decision

    def route_for_prompt(self, prompt: str) -> RoutingDecision:
        """Infer task category from a prompt and route accordingly.

        Uses keyword matching — no LLM call.
        """
        category = self._classify_prompt(prompt)
        return self.route(category)

    def benchmark_all(
        self,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, List]:
        """Run benchmark tasks against all (or specified) models.

        Results are automatically stored in SQLite for future scoring.
        """
        if not self._profiles:
            self.discover()

        self.client.start_daemon(timeout_seconds=5.0)

        targets = self._profiles
        if model_names:
            name_set = {n.lower() for n in model_names}
            targets = [p for p in self._profiles if p.name.lower() in name_set]

        all_results: Dict[str, List] = {}
        for profile in targets:
            results = run_benchmarks(
                client=self.client,
                model_name=profile.name,
                storage=self.storage,
            )
            all_results[profile.name] = results

        return all_results

    def benchmark_model(self, model_name: str) -> List:
        """Run benchmarks for a single model."""
        self.client.start_daemon(timeout_seconds=5.0)
        return run_benchmarks(
            client=self.client,
            model_name=model_name,
            storage=self.storage,
        )

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def profiles(self) -> List[ModelProfile]:
        """Get discovered model profiles."""
        if not self._profiles:
            self.discover()
        return self._profiles

    @property
    def hardware(self) -> HardwareInfo:
        """Get local hardware info."""
        if self._hardware is None:
            self._hardware = get_hardware_info()
        return self._hardware

    def get_profile(self, model_name: str) -> Optional[ModelProfile]:
        """Get a specific model's profile."""
        for p in self.profiles:
            if p.name.lower() == model_name.lower():
                return p
            if p.name.split(":")[0].lower() == model_name.lower():
                return p
        return None

    def save_config(self) -> None:
        """Persist router config to disk."""
        self.config.save_to_file(self._config_path)
        logger.info("Router config saved to %s", self._config_path)

    def reload_config(self) -> None:
        """Reload router config from disk."""
        self.config = RouterConfig.load_from_file(self._config_path)
        self.scorer = ModelScorer(config=self.config, storage=self.storage)
        logger.info("Router config reloaded from %s", self._config_path)

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _classify_prompt(prompt: str) -> TaskCategory:
        """Classify a prompt into a task category using keyword matching.

        Deterministic — no LLM call.
        """
        lower = prompt.lower()

        # Coding keywords
        code_keywords = {
            "write a function", "implement", "code", "debug", "fix the bug",
            "refactor", "optimize", "class", "method", "api", "endpoint",
            "unittest", "test case", "def ", "import ", "function",
            "compile", "syntax", "error in code",
        }
        if any(kw in lower for kw in code_keywords):
            if any(kw in lower for kw in {"debug", "fix the bug", "error in code"}):
                return TaskCategory.DEBUGGING
            if any(kw in lower for kw in {"review", "refactor"}):
                return TaskCategory.CODE_REVIEW
            return TaskCategory.CODE_GENERATION

        # Vision keywords
        vision_keywords = {
            "image", "screenshot", "photo", "picture", "visual",
            "what do you see", "describe this", "ui design",
        }
        if any(kw in lower for kw in vision_keywords):
            return TaskCategory.IMAGE_ANALYSIS

        # Planning keywords
        plan_keywords = {
            "plan", "break down", "decompose", "steps to", "roadmap",
            "architecture", "design", "strategy",
        }
        if any(kw in lower for kw in plan_keywords):
            return TaskCategory.PLANNING

        # Reasoning keywords
        reason_keywords = {
            "why", "explain", "analyze", "compare", "evaluate",
            "calculate", "prove", "logic", "reasoning",
        }
        if any(kw in lower for kw in reason_keywords):
            return TaskCategory.REASONING

        # Summarization keywords
        summary_keywords = {
            "summarize", "summary", "tldr", "brief", "condense",
            "key points", "main ideas",
        }
        if any(kw in lower for kw in summary_keywords):
            return TaskCategory.SUMMARIZATION

        # Default to chat
        return TaskCategory.CHAT
