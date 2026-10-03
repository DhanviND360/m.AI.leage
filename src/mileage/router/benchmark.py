"""Standardized local benchmark tasks for model evaluation.

Each benchmark is a small, deterministic task that can run in seconds.
Results are stored in SQLite and used by the scorer to refine routing.
"""

import json
import time
from typing import List, Optional

from mileage.core.logger import logger
from mileage.models.ollama_client import OllamaClient
from mileage.router.schemas import (
    BenchmarkResult,
    BenchmarkTask,
    TaskCategory,
)
from mileage.router.storage import RouterStorage


# ── Standard Benchmark Suite ─────────────────────────────────────────────────

BENCHMARK_TASKS: List[BenchmarkTask] = [
    # Coding
    BenchmarkTask(
        task_id="bench_code_fizzbuzz",
        name="FizzBuzz Function",
        category=TaskCategory.CODE_GENERATION,
        prompt=(
            "Write a Python function called fizzbuzz(n) that returns a list of strings "
            "from 1 to n. For multiples of 3 use 'Fizz', multiples of 5 use 'Buzz', "
            "multiples of both use 'FizzBuzz', otherwise the number as a string. "
            "Output ONLY the function, no explanation."
        ),
        expected_contains=["def fizzbuzz", "Fizz", "Buzz"],
        max_tokens=256,
        timeout_seconds=20.0,
    ),
    BenchmarkTask(
        task_id="bench_code_reverse",
        name="String Reversal",
        category=TaskCategory.CODE_GENERATION,
        prompt=(
            "Write a Python function called reverse_string(s) that reverses a string "
            "without using slicing or the reversed() built-in. Output ONLY the function."
        ),
        expected_contains=["def reverse_string"],
        max_tokens=200,
        timeout_seconds=15.0,
    ),

    # Reasoning
    BenchmarkTask(
        task_id="bench_reason_math",
        name="Basic Math Reasoning",
        category=TaskCategory.REASONING,
        prompt=(
            "A train travels 120 km in 2 hours. It then travels 90 km in 1.5 hours. "
            "What is the average speed for the entire journey? "
            "Answer with just the number in km/h."
        ),
        expected_contains=["60"],
        max_tokens=100,
        timeout_seconds=15.0,
    ),
    BenchmarkTask(
        task_id="bench_reason_logic",
        name="Logic Puzzle",
        category=TaskCategory.REASONING,
        prompt=(
            "If all roses are flowers, and some flowers fade quickly, "
            "can we conclude that some roses fade quickly? "
            "Answer 'No' and explain in one sentence why."
        ),
        expected_contains=["No"],
        max_tokens=100,
        timeout_seconds=15.0,
    ),

    # Chat / Instruction
    BenchmarkTask(
        task_id="bench_chat_greeting",
        name="Greeting Response",
        category=TaskCategory.CHAT,
        prompt="Say hello and introduce yourself in exactly one sentence.",
        expected_contains=[],  # Any response is valid
        max_tokens=100,
        timeout_seconds=10.0,
    ),

    # Summarization
    BenchmarkTask(
        task_id="bench_summarize",
        name="Text Summarization",
        category=TaskCategory.SUMMARIZATION,
        prompt=(
            "Summarize in one sentence: 'The Python programming language was created by "
            "Guido van Rossum and first released in 1991. It emphasizes code readability "
            "and supports multiple programming paradigms including procedural, "
            "object-oriented, and functional programming.'"
        ),
        expected_contains=["Python"],
        max_tokens=100,
        timeout_seconds=15.0,
    ),

    # Planning
    BenchmarkTask(
        task_id="bench_plan_todo",
        name="Task Decomposition",
        category=TaskCategory.PLANNING,
        prompt=(
            "Break down the task 'Build a REST API for a todo app' into exactly 5 steps. "
            "Number each step 1-5. Be concise — one line per step."
        ),
        expected_contains=["1", "2", "3"],
        max_tokens=256,
        timeout_seconds=20.0,
    ),
]


def get_benchmark_tasks(
    category: Optional[TaskCategory] = None,
) -> List[BenchmarkTask]:
    """Get benchmark tasks, optionally filtered by category."""
    if category:
        return [t for t in BENCHMARK_TASKS if t.category == category]
    return list(BENCHMARK_TASKS)


def run_single_benchmark(
    client: OllamaClient,
    model_name: str,
    task: BenchmarkTask,
) -> BenchmarkResult:
    """Run a single benchmark task against a model.

    Returns a BenchmarkResult with latency, success, and token metrics.
    """
    start = time.perf_counter()
    success = False
    response_text = ""
    tokens_generated = 0
    error_msg = None

    try:
        with client._get_http_client(timeout=task.timeout_seconds) as http:
            resp = http.post(
                "/api/generate",
                json={
                    "model": model_name,
                    "prompt": task.prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": task.max_tokens,
                    },
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("response", "")
                tokens_generated = data.get("eval_count", max(1, len(response_text) // 4))

                # Check expected substrings
                if task.expected_contains:
                    response_lower = response_text.lower()
                    matches = sum(
                        1 for exp in task.expected_contains
                        if exp.lower() in response_lower
                    )
                    success = matches >= len(task.expected_contains)
                else:
                    # No expected content — pass if we got any non-empty response
                    success = len(response_text.strip()) > 0
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:100]}"

    except Exception as e:
        error_msg = str(e)[:200]

    elapsed_ms = (time.perf_counter() - start) * 1000
    tps = (tokens_generated / (elapsed_ms / 1000)) if elapsed_ms > 0 and tokens_generated > 0 else 0.0

    return BenchmarkResult(
        model_name=model_name,
        task_id=task.task_id,
        success=success,
        latency_ms=round(elapsed_ms, 2),
        tokens_generated=tokens_generated,
        tokens_per_sec=round(tps, 2),
        response_preview=response_text[:200],
        error=error_msg,
    )


def run_benchmarks(
    client: OllamaClient,
    model_name: str,
    storage: Optional[RouterStorage] = None,
    categories: Optional[List[TaskCategory]] = None,
) -> List[BenchmarkResult]:
    """Run all (or filtered) benchmark tasks against a model.

    Results are automatically saved to SQLite if storage is provided.
    """
    tasks = BENCHMARK_TASKS
    if categories:
        cat_set = set(categories)
        tasks = [t for t in tasks if t.category in cat_set]

    results: List[BenchmarkResult] = []

    for task in tasks:
        logger.info("Benchmarking '%s' on task '%s'...", model_name, task.name)
        result = run_single_benchmark(client, model_name, task)
        results.append(result)

        if storage:
            storage.save_benchmark(result)

        logger.info(
            "  → %s | %.0fms | %d tok | %.1f tok/s",
            "PASS" if result.success else "FAIL",
            result.latency_ms,
            result.tokens_generated,
            result.tokens_per_sec,
        )

    return results
