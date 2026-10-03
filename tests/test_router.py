"""Tests for the Model Router subsystem.

Covers:
- Schema validation (ModelProfile, RoutingDecision, RouterConfig)
- Capability registry (heuristic scoring, size adjustment)
- SQLite storage (benchmarks, decisions, profiles)
- Deterministic scorer (capability match, size bonus, confidence)
- Router orchestrator (discovery, routing, prompt classification)
- Benchmark task definitions
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mileage.router.schemas import (
    BenchmarkResult,
    BenchmarkTask,
    Capability,
    CapabilityScore,
    HardwareInfo,
    ModelProfile,
    RouterConfig,
    RoutingDecision,
    TaskCategory,
    TaskRule,
)
from mileage.router.registry import (
    populate_capabilities,
    populate_all,
    _adjust_for_size,
)
from mileage.router.storage import RouterStorage
from mileage.router.scorer import ModelScorer
from mileage.router.benchmark import get_benchmark_tasks, BENCHMARK_TASKS
from mileage.router.router import ModelRouter


# ── Test Fixtures ────────────────────────────────────────────────────────────

def _make_profile(
    name: str = "gemma3:4b",
    family: str = "gemma3",
    param_size: str = "4B",
    size_bytes: int = 3_000_000_000,
    context_length: int = 8192,
    caps: dict | None = None,
) -> ModelProfile:
    """Create a test ModelProfile."""
    profile = ModelProfile(
        name=name,
        family=family,
        parameter_size=param_size,
        size_bytes=size_bytes,
        size_human=f"{size_bytes / 1e9:.1f} GB",
        context_length=context_length,
    )
    if caps:
        profile.capabilities = [
            CapabilityScore(capability=Capability(k), score=v, source="test")
            for k, v in caps.items()
        ]
    return profile


def _make_storage() -> RouterStorage:
    """Create a temporary RouterStorage."""
    tmpdir = tempfile.mkdtemp()
    return RouterStorage(db_path=Path(tmpdir) / "test_router.db")


# ── Schema Tests ─────────────────────────────────────────────────────────────


class TestSchemas:
    """Validate Pydantic schemas for the router."""

    def test_model_profile_defaults(self):
        p = ModelProfile(name="test:latest")
        assert p.family == "unknown"
        assert p.context_length == 4096
        assert p.capabilities == []
        assert p.get_capability_score(Capability.CODING) == 0.0

    def test_model_profile_capability_access(self):
        p = _make_profile(caps={"coding": 0.8, "reasoning": 0.6})
        assert p.get_capability_score(Capability.CODING) == 0.8
        assert p.has_capability(Capability.CODING, threshold=0.5)
        assert not p.has_capability(Capability.VISION, threshold=0.3)

    def test_model_profile_parameter_parsing(self):
        p = _make_profile(param_size="4B")
        assert p.parameter_count_billions == 4.0

        p2 = _make_profile(param_size="700M")
        assert p2.parameter_count_billions == 0.7

        p3 = _make_profile(param_size=None)
        assert p3.parameter_count_billions is None

    def test_routing_decision_fields(self):
        d = RoutingDecision(
            selected_model="gemma3:4b",
            reason="Best capability match for coding",
            task_category=TaskCategory.CODE_GENERATION,
            confidence=0.85,
            scores={"gemma3:4b": 0.85, "llama3:8b": 0.72},
        )
        assert d.selected_model == "gemma3:4b"
        assert d.confidence == 0.85
        assert len(d.scores) == 2

    def test_router_config_defaults(self):
        config = RouterConfig()
        assert config.confidence_threshold == 0.4
        assert config.prefer_smallest is True
        assert len(config.task_rules) > 0
        assert len(config.capability_weights) > 0

    def test_router_config_get_rule(self):
        config = RouterConfig()
        rule = config.get_rule(TaskCategory.CODE_GENERATION)
        assert Capability.CODING in rule.required_capabilities

        unknown_rule = config.get_rule(TaskCategory.GENERAL)
        assert unknown_rule.category == TaskCategory.GENERAL

    def test_router_config_save_load(self):
        config = RouterConfig(confidence_threshold=0.7, prefer_smallest=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)
        config.save_to_file(path)
        loaded = RouterConfig.load_from_file(path)
        assert loaded.confidence_threshold == 0.7
        assert loaded.prefer_smallest is False

    def test_benchmark_task_structure(self):
        task = BenchmarkTask(
            task_id="test_1",
            name="Test Task",
            category=TaskCategory.GENERAL,
            prompt="Hello",
            expected_contains=["world"],
        )
        assert task.max_tokens == 256
        assert task.timeout_seconds == 30.0

    def test_hardware_info(self):
        hw = HardwareInfo(gpu_available=True, gpu_name="Test GPU", cpu_cores=8)
        assert hw.gpu_available
        assert hw.cpu_cores == 8


# ── Registry Tests ───────────────────────────────────────────────────────────


class TestRegistry:
    """Validate capability registry heuristics."""

    def test_gemma3_capabilities(self):
        p = _make_profile(name="gemma3:4b", family="gemma3")
        populate_capabilities(p)
        assert p.get_capability_score(Capability.CODING) > 0.5
        assert p.get_capability_score(Capability.VISION) > 0.3
        assert len(p.capabilities) > 0

    def test_codellama_specialization(self):
        p = _make_profile(name="codellama:7b", family="codellama", param_size="7B")
        populate_capabilities(p)
        assert p.get_capability_score(Capability.CODING) >= 0.8
        # Coding should be much higher than chat
        assert p.get_capability_score(Capability.CODING) > p.get_capability_score(Capability.CHAT)

    def test_unknown_model_gets_defaults(self):
        p = _make_profile(name="mystery-model:latest", family="unknown_family")
        populate_capabilities(p)
        assert len(p.capabilities) > 0
        assert p.get_capability_score(Capability.CHAT) > 0

    def test_size_adjustment_small(self):
        caps = {Capability.CODING: 0.7, Capability.SPEED: 0.5}
        adjusted = _adjust_for_size(caps, parameter_billions=1.0)
        assert adjusted[Capability.SPEED] > 0.5  # Speed bonus for small model

    def test_size_adjustment_large(self):
        caps = {Capability.CODING: 0.7, Capability.SPEED: 0.5}
        adjusted = _adjust_for_size(caps, parameter_billions=70.0)
        assert adjusted[Capability.SPEED] < 0.5  # Speed penalty for huge model

    def test_populate_all(self):
        profiles = [
            _make_profile(name="gemma3:4b", family="gemma3"),
            _make_profile(name="llama3:8b", family="llama3", param_size="8B"),
        ]
        populate_all(profiles)
        for p in profiles:
            assert len(p.capabilities) > 0

    def test_vision_model_detection(self):
        p = _make_profile(name="llava:7b", family="llava")
        populate_capabilities(p)
        assert p.get_capability_score(Capability.VISION) > 0.5


# ── Storage Tests ────────────────────────────────────────────────────────────


class TestStorage:
    """Validate SQLite storage operations."""

    def test_save_and_get_benchmark(self):
        storage = _make_storage()
        result = BenchmarkResult(
            model_name="test:latest",
            task_id="test_task",
            success=True,
            latency_ms=500.0,
            tokens_generated=100,
            tokens_per_sec=200.0,
        )
        storage.save_benchmark(result)
        results = storage.get_model_benchmarks("test:latest")
        assert len(results) == 1
        assert results[0].success is True
        storage.close()

    def test_model_stats(self):
        storage = _make_storage()
        for i in range(5):
            storage.save_benchmark(BenchmarkResult(
                model_name="stats_model",
                task_id=f"task_{i}",
                success=i < 4,  # 4 pass, 1 fail
                latency_ms=100.0 * (i + 1),
                tokens_generated=50,
                tokens_per_sec=50.0,
            ))
        stats = storage.get_model_stats("stats_model")
        assert stats["total"] == 5
        assert stats["success_rate"] == 0.8
        assert stats["avg_latency_ms"] > 0
        storage.close()

    def test_save_and_get_decision(self):
        storage = _make_storage()
        decision = RoutingDecision(
            selected_model="gemma3:4b",
            reason="Best match",
            task_category=TaskCategory.CODE_GENERATION,
            confidence=0.8,
            scores={"gemma3:4b": 0.8},
        )
        storage.save_decision(decision)
        decisions = storage.get_recent_decisions(limit=5)
        assert len(decisions) == 1
        assert decisions[0]["selected_model"] == "gemma3:4b"
        storage.close()

    def test_save_profile(self):
        storage = _make_storage()
        p = _make_profile(caps={"coding": 0.8})
        storage.save_profile(p)
        profiles = storage.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0]["name"] == "gemma3:4b"
        storage.close()

    def test_totals(self):
        storage = _make_storage()
        assert storage.get_total_benchmarks() == 0
        assert storage.get_total_decisions() == 0
        storage.save_benchmark(BenchmarkResult(
            model_name="test", task_id="t1", success=True, latency_ms=100,
        ))
        assert storage.get_total_benchmarks() == 1
        storage.close()


# ── Scorer Tests ─────────────────────────────────────────────────────────────


class TestScorer:
    """Validate deterministic scoring engine."""

    def test_capability_scoring(self):
        config = RouterConfig()
        scorer = ModelScorer(config=config)

        good_coder = _make_profile(
            name="coder:7b", family="codellama", param_size="7B",
            caps={"coding": 0.9, "reasoning": 0.5, "chat": 0.4},
        )
        score, reason = scorer.score_model(good_coder, TaskCategory.CODE_GENERATION)
        assert score > 0.5
        assert "coder:7b" in reason

    def test_missing_required_capability_penalized(self):
        config = RouterConfig()
        scorer = ModelScorer(config=config)

        # Model without coding → should score low for code_generation
        chat_only = _make_profile(
            name="chatty:1b", caps={"chat": 0.8, "speed": 0.9},
        )
        score, _ = scorer.score_model(chat_only, TaskCategory.CODE_GENERATION)

        # Model with coding → should score higher
        coder = _make_profile(
            name="coder:7b", caps={"coding": 0.8, "reasoning": 0.6},
        )
        code_score, _ = scorer.score_model(coder, TaskCategory.CODE_GENERATION)

        assert code_score > score

    def test_score_all_sorts_descending(self):
        config = RouterConfig()
        scorer = ModelScorer(config=config)

        profiles = [
            _make_profile(name="weak:1b", caps={"coding": 0.2, "chat": 0.4}),
            _make_profile(name="strong:7b", caps={"coding": 0.9, "reasoning": 0.8}),
            _make_profile(name="mid:3b", caps={"coding": 0.6, "reasoning": 0.5}),
        ]
        scored = scorer.score_all(profiles, TaskCategory.CODE_GENERATION)
        scores = [s for _, s, _ in scored]
        assert scores == sorted(scores, reverse=True)

    def test_select_best_returns_decision(self):
        config = RouterConfig()
        scorer = ModelScorer(config=config)

        profiles = [
            _make_profile(name="model_a", caps={"coding": 0.9, "reasoning": 0.7}),
            _make_profile(name="model_b", caps={"coding": 0.5, "chat": 0.8}),
        ]
        decision = scorer.select_best(profiles, TaskCategory.CODE_GENERATION)
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_model in ["model_a", "model_b"]
        assert decision.confidence > 0
        assert len(decision.scores) == 2

    def test_confidence_threshold_uses_fallback(self):
        config = RouterConfig(confidence_threshold=0.95, fallback_model="fallback:latest")
        scorer = ModelScorer(config=config)

        profiles = [
            _make_profile(name="weak:1b", caps={"coding": 0.3, "chat": 0.3}),
        ]
        decision = scorer.select_best(profiles, TaskCategory.CODE_GENERATION)
        # Score should be below 0.95, so fallback should be used
        assert decision.selected_model == "fallback:latest"
        assert "Low confidence" in decision.reason

    def test_size_bonus_prefers_smaller(self):
        config = RouterConfig(prefer_smallest=True)
        scorer = ModelScorer(config=config)

        small = _make_profile(
            name="small:2b", param_size="2B",
            caps={"coding": 0.7, "reasoning": 0.6, "chat": 0.7},
        )
        big = _make_profile(
            name="big:70b", param_size="70B",
            caps={"coding": 0.7, "reasoning": 0.6, "chat": 0.7},
        )
        small_score, _ = scorer.score_model(small, TaskCategory.CHAT)
        big_score, _ = scorer.score_model(big, TaskCategory.CHAT)
        # Small should get a bonus
        assert small_score >= big_score

    def test_benchmark_history_influences_score(self):
        storage = _make_storage()
        # Add benchmark data
        for i in range(3):
            storage.save_benchmark(BenchmarkResult(
                model_name="bench_model",
                task_id=f"task_{i}",
                success=True,
                latency_ms=200.0,
                tokens_generated=100,
                tokens_per_sec=500.0,
            ))

        config = RouterConfig(enable_benchmarks=True, benchmark_weight=0.3)
        scorer = ModelScorer(config=config, storage=storage)

        profile = _make_profile(
            name="bench_model", caps={"chat": 0.7},
        )
        score_with_bench, _ = scorer.score_model(profile, TaskCategory.CHAT)

        # Without benchmarks
        scorer_no_bench = ModelScorer(
            config=RouterConfig(enable_benchmarks=False), storage=None,
        )
        score_without, _ = scorer_no_bench.score_model(profile, TaskCategory.CHAT)

        # Benchmark history should influence the score
        assert score_with_bench != score_without
        storage.close()

    def test_empty_profiles_returns_fallback(self):
        config = RouterConfig(fallback_model="emergency:latest")
        scorer = ModelScorer(config=config)
        decision = scorer.select_best([], TaskCategory.GENERAL)
        assert decision.selected_model == "emergency:latest"
        assert decision.confidence == 0.0


# ── Router Orchestrator Tests ────────────────────────────────────────────────


class TestRouter:
    """Validate the main ModelRouter orchestrator."""

    def _make_mock_client(self, models: list[dict]) -> MagicMock:
        """Create a mock OllamaClient with specified models."""
        from mileage.models.schemas import ModelInfo, ModelDetails

        mock = MagicMock()
        mock.start_daemon.return_value = True
        mock.list_models.return_value = [
            ModelInfo(
                name=m["name"],
                model=m["name"],
                size_bytes=m.get("size", 3_000_000_000),
                size_human=f"{m.get('size', 3e9) / 1e9:.1f} GB",
                details=ModelDetails(
                    family=m.get("family", "unknown"),
                    parameter_size=m.get("params", "4B"),
                ),
            )
            for m in models
        ]

        # Mock HTTP client for context length fetch
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404  # Skip actual context length fetch
        mock_http.post.return_value = mock_resp
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock._get_http_client.return_value = mock_http

        return mock

    def test_discover_populates_profiles(self):
        mock = self._make_mock_client([
            {"name": "gemma3:4b", "family": "gemma3", "params": "4B"},
            {"name": "llama3:8b", "family": "llama3", "params": "8B"},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            router = ModelRouter(
                ollama_client=mock,
                workspace_root=Path(tmpdir),
            )
            profiles = router.discover()
            assert len(profiles) == 2
            for p in profiles:
                assert len(p.capabilities) > 0
            router.storage.close()

    def test_route_selects_model(self):
        mock = self._make_mock_client([
            {"name": "gemma3:4b", "family": "gemma3", "params": "4B"},
            {"name": "codellama:7b", "family": "codellama", "params": "7B"},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            router = ModelRouter(
                ollama_client=mock,
                workspace_root=Path(tmpdir),
            )
            decision = router.route(TaskCategory.CODE_GENERATION)
            assert decision.selected_model in ["gemma3:4b", "codellama:7b"]
            assert decision.confidence > 0
            assert decision.reason != ""
            router.storage.close()

    def test_route_for_prompt_classifies(self):
        mock = self._make_mock_client([
            {"name": "gemma3:4b", "family": "gemma3", "params": "4B"},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            router = ModelRouter(
                ollama_client=mock,
                workspace_root=Path(tmpdir),
            )
            decision = router.route_for_prompt("Write a function to sort a list")
            assert decision.task_category == TaskCategory.CODE_GENERATION
            router.storage.close()

    def test_prompt_classification(self):
        assert ModelRouter._classify_prompt("debug the login bug") == TaskCategory.DEBUGGING
        assert ModelRouter._classify_prompt("write a function") == TaskCategory.CODE_GENERATION
        assert ModelRouter._classify_prompt("summarize this document") == TaskCategory.SUMMARIZATION
        assert ModelRouter._classify_prompt("plan the architecture") == TaskCategory.PLANNING
        assert ModelRouter._classify_prompt("analyze this screenshot") == TaskCategory.IMAGE_ANALYSIS
        assert ModelRouter._classify_prompt("why does gravity work") == TaskCategory.REASONING
        assert ModelRouter._classify_prompt("hello how are you") == TaskCategory.CHAT

    def test_config_persistence(self):
        mock = self._make_mock_client([{"name": "test:1b", "family": "test"}])
        with tempfile.TemporaryDirectory() as tmpdir:
            router = ModelRouter(
                ollama_client=mock,
                workspace_root=Path(tmpdir),
            )
            router.config.confidence_threshold = 0.99
            router.save_config()
            config_path = Path(tmpdir) / ".mileage" / "router_config.json"
            assert config_path.is_file()
            loaded = RouterConfig.load_from_file(config_path)
            assert loaded.confidence_threshold == 0.99
            router.storage.close()

    def test_single_model_always_selected(self):
        mock = self._make_mock_client([
            {"name": "only_model:4b", "family": "gemma3", "params": "4B"},
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            router = ModelRouter(
                ollama_client=mock,
                workspace_root=Path(tmpdir),
            )
            decision = router.route(TaskCategory.GENERAL)
            assert decision.selected_model == "only_model:4b"
            router.storage.close()


# ── Benchmark Definition Tests ───────────────────────────────────────────────


class TestBenchmarkDefinitions:
    """Validate benchmark task definitions."""

    def test_all_tasks_have_ids(self):
        for task in BENCHMARK_TASKS:
            assert task.task_id.startswith("bench_")
            assert task.name != ""
            assert task.prompt != ""

    def test_filter_by_category(self):
        coding = get_benchmark_tasks(TaskCategory.CODE_GENERATION)
        assert len(coding) >= 1
        assert all(t.category == TaskCategory.CODE_GENERATION for t in coding)

        reasoning = get_benchmark_tasks(TaskCategory.REASONING)
        assert len(reasoning) >= 1

    def test_all_categories_covered(self):
        categories = {t.category for t in BENCHMARK_TASKS}
        assert TaskCategory.CODE_GENERATION in categories
        assert TaskCategory.REASONING in categories
        assert TaskCategory.CHAT in categories
        assert TaskCategory.SUMMARIZATION in categories
        assert TaskCategory.PLANNING in categories
