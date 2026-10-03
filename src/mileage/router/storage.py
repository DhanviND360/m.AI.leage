"""SQLite storage for benchmark results, routing history, and model profiles.

All data is stored locally in .mileage/router.db — no cloud dependency.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mileage.core.logger import logger
from mileage.router.schemas import BenchmarkResult, ModelProfile, RoutingDecision


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    success     INTEGER NOT NULL,
    latency_ms  REAL NOT NULL,
    tokens_generated INTEGER DEFAULT 0,
    tokens_per_sec   REAL DEFAULT 0.0,
    response_preview TEXT DEFAULT '',
    error       TEXT,
    run_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    selected_model      TEXT NOT NULL,
    fallback_model      TEXT,
    reason              TEXT NOT NULL,
    task_category       TEXT NOT NULL,
    required_capabilities TEXT DEFAULT '[]',
    capability_match    REAL DEFAULT 0.0,
    confidence          REAL DEFAULT 0.5,
    scores_json         TEXT DEFAULT '{}',
    decided_at          TEXT NOT NULL,
    decision_time_ms    REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS model_profiles (
    name            TEXT PRIMARY KEY,
    family          TEXT DEFAULT 'unknown',
    parameter_size  TEXT,
    quantization    TEXT,
    size_bytes      INTEGER DEFAULT 0,
    context_length  INTEGER DEFAULT 4096,
    capabilities_json TEXT DEFAULT '[]',
    avg_latency_ms  REAL,
    avg_tokens_per_sec REAL,
    benchmark_success_rate REAL,
    benchmark_run_count INTEGER DEFAULT 0,
    discovered_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bench_model ON benchmark_results(model_name);
CREATE INDEX IF NOT EXISTS idx_bench_task  ON benchmark_results(task_id);
CREATE INDEX IF NOT EXISTS idx_route_model ON routing_decisions(selected_model);
"""


class RouterStorage:
    """Local SQLite storage for router state and history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(".mileage") / "router.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        try:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        except Exception as e:
            logger.error("Failed to initialize router database: %s", e)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Benchmark Results ────────────────────────────────────────────────

    def save_benchmark(self, result: BenchmarkResult) -> None:
        """Store a benchmark result."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO benchmark_results
               (model_name, task_id, success, latency_ms, tokens_generated,
                tokens_per_sec, response_preview, error, run_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.model_name,
                result.task_id,
                int(result.success),
                result.latency_ms,
                result.tokens_generated,
                result.tokens_per_sec,
                result.response_preview[:200],
                result.error,
                result.run_at,
            ),
        )
        conn.commit()

    def get_model_benchmarks(self, model_name: str) -> List[BenchmarkResult]:
        """Get all benchmark results for a model."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM benchmark_results WHERE model_name = ? ORDER BY run_at DESC",
            (model_name,),
        ).fetchall()
        return [
            BenchmarkResult(
                model_name=r["model_name"],
                task_id=r["task_id"],
                success=bool(r["success"]),
                latency_ms=r["latency_ms"],
                tokens_generated=r["tokens_generated"],
                tokens_per_sec=r["tokens_per_sec"],
                response_preview=r["response_preview"],
                error=r["error"],
                run_at=r["run_at"],
            )
            for r in rows
        ]

    def get_model_stats(self, model_name: str) -> Dict[str, float]:
        """Get aggregated benchmark stats for a model."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                AVG(latency_ms) as avg_latency,
                AVG(tokens_per_sec) as avg_tps
               FROM benchmark_results WHERE model_name = ?""",
            (model_name,),
        ).fetchone()

        if not row or row["total"] == 0:
            return {"total": 0, "success_rate": 0.0, "avg_latency_ms": 0.0, "avg_tps": 0.0}

        return {
            "total": row["total"],
            "success_rate": row["successes"] / row["total"] if row["total"] else 0.0,
            "avg_latency_ms": row["avg_latency"] or 0.0,
            "avg_tps": row["avg_tps"] or 0.0,
        }

    # ── Routing History ──────────────────────────────────────────────────

    def save_decision(self, decision: RoutingDecision) -> None:
        """Store a routing decision for audit trail."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO routing_decisions
               (selected_model, fallback_model, reason, task_category,
                required_capabilities, capability_match, confidence,
                scores_json, decided_at, decision_time_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision.selected_model,
                decision.fallback_model,
                decision.reason,
                decision.task_category.value,
                json.dumps([c.value for c in decision.required_capabilities]),
                decision.capability_match_score,
                decision.confidence,
                json.dumps(decision.scores),
                decision.decided_at,
                decision.decision_time_ms,
            ),
        )
        conn.commit()

    def get_recent_decisions(self, limit: int = 20) -> List[Dict]:
        """Get recent routing decisions."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM routing_decisions ORDER BY decided_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Model Profiles (Cache) ───────────────────────────────────────────

    def save_profile(self, profile: ModelProfile) -> None:
        """Upsert a model profile."""
        conn = self._get_conn()
        caps_json = json.dumps(
            [{"capability": cs.capability.value, "score": cs.score, "source": cs.source}
             for cs in profile.capabilities]
        )
        conn.execute(
            """INSERT OR REPLACE INTO model_profiles
               (name, family, parameter_size, quantization, size_bytes,
                context_length, capabilities_json, avg_latency_ms,
                avg_tokens_per_sec, benchmark_success_rate,
                benchmark_run_count, discovered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                profile.name,
                profile.family,
                profile.parameter_size,
                profile.quantization,
                profile.size_bytes,
                profile.context_length,
                caps_json,
                profile.avg_latency_ms,
                profile.avg_tokens_per_sec,
                profile.benchmark_success_rate,
                profile.benchmark_run_count,
                profile.discovered_at,
            ),
        )
        conn.commit()

    def get_all_profiles(self) -> List[Dict]:
        """Get all cached model profiles."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM model_profiles ORDER BY size_bytes ASC").fetchall()
        return [dict(r) for r in rows]

    def get_total_benchmarks(self) -> int:
        """Get total number of benchmark runs."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM benchmark_results").fetchone()
        return row["cnt"] if row else 0

    def get_total_decisions(self) -> int:
        """Get total number of routing decisions."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM routing_decisions").fetchone()
        return row["cnt"] if row else 0
