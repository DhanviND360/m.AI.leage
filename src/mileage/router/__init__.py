"""Model Router subsystem for m.AI.leage.

Automatic model discovery, capability scoring, benchmarking,
and deterministic routing for local Ollama models.
"""

from mileage.router.router import ModelRouter
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
from mileage.router.storage import RouterStorage
from mileage.router.discovery import discover_models, get_hardware_info
from mileage.router.registry import populate_capabilities, populate_all
from mileage.router.benchmark import run_benchmarks, get_benchmark_tasks
from mileage.router.scorer import ModelScorer

__all__ = [
    "ModelRouter",
    "ModelScorer",
    "RouterStorage",
    "RouterConfig",
    "ModelProfile",
    "RoutingDecision",
    "Capability",
    "CapabilityScore",
    "TaskCategory",
    "TaskRule",
    "BenchmarkResult",
    "BenchmarkTask",
    "HardwareInfo",
    "discover_models",
    "get_hardware_info",
    "populate_capabilities",
    "populate_all",
    "run_benchmarks",
    "get_benchmark_tasks",
]
