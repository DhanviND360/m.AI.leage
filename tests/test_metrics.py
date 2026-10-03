"""Tests for local metrics logging and aggregation."""

from pathlib import Path
from mileage.metrics.schemas import MetricRecord
from mileage.metrics.tracker import MetricsTracker


def test_metrics_record_and_summary(tmp_path: Path):
    metrics_file = tmp_path / "metrics.jsonl"
    tracker = MetricsTracker(metrics_file)

    assert tracker.get_summary().total_runs == 0

    # Record two runs
    tracker.record(
        MetricRecord(
            command="run",
            model="llama3.2",
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            latency_ms=120.5,
            success=True,
        )
    )
    tracker.record(
        MetricRecord(
            command="run",
            model="mistral",
            prompt_tokens=40,
            completion_tokens=60,
            total_tokens=100,
            latency_ms=80.0,
            success=False,
            error_type="OllamaConnectionError",
        )
    )

    records = tracker.load_all()
    assert len(records) == 2
    assert records[0].model == "llama3.2"
    assert records[1].model == "mistral"

    summary = tracker.get_summary()
    assert summary.total_runs == 2
    assert summary.successful_runs == 1
    assert summary.failed_runs == 1
    assert summary.success_rate_percent == 50.0
    assert summary.total_tokens == 250
    assert summary.total_prompt_tokens == 90
    assert summary.total_completion_tokens == 160
    assert summary.average_latency_ms == round((120.5 + 80.0) / 2, 1)
    assert summary.model_usage_counts == {"llama3.2": 1, "mistral": 1}
