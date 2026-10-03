"""Local metrics tracker writing JSONL records to workspace."""

import json
from pathlib import Path
from typing import List, Optional

from mileage.core.logger import logger
from mileage.metrics.schemas import MetricRecord, MetricsSummary


class MetricsTracker:
    """Manages recording and summarizing local metrics."""

    def __init__(self, metrics_file: Optional[Path] = None):
        self.metrics_file = metrics_file

    def record(self, record: MetricRecord) -> None:
        """Append a metric record to the local JSONL metrics file."""
        if not self.metrics_file:
            return

        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception as e:
            logger.warning("Failed to record metric locally: %s", e)

    def load_all(self) -> List[MetricRecord]:
        """Read all metric records from the local metrics file."""
        if not self.metrics_file or not self.metrics_file.is_file():
            return []

        records: List[MetricRecord] = []
        try:
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    clean = line.strip()
                    if clean:
                        try:
                            records.append(MetricRecord.model_validate_json(clean))
                        except Exception:
                            continue
        except Exception as e:
            logger.warning("Failed reading metrics file: %s", e)

        return records

    def get_summary(self) -> MetricsSummary:
        """Compute aggregated statistics across all local runs."""
        records = self.load_all()
        if not records:
            return MetricsSummary()

        total_runs = len(records)
        successful_runs = sum(1 for r in records if r.success)
        failed_runs = total_runs - successful_runs
        success_rate = (successful_runs / total_runs * 100.0) if total_runs > 0 else 0.0

        total_prompt = sum(r.prompt_tokens for r in records)
        total_comp = sum(r.completion_tokens for r in records)
        total_tok = sum(r.total_tokens for r in records)
        avg_lat = sum(r.latency_ms for r in records) / total_runs if total_runs > 0 else 0.0

        usage_counts = {}
        for r in records:
            m = r.model or "unknown"
            usage_counts[m] = usage_counts.get(m, 0) + 1

        return MetricsSummary(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            success_rate_percent=round(success_rate, 1),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_comp,
            total_tokens=total_tok,
            average_latency_ms=round(avg_lat, 1),
            model_usage_counts=usage_counts,
            first_run_at=records[0].timestamp if records else None,
            last_run_at=records[-1].timestamp if records else None,
        )
