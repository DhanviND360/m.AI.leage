"""Tests for m.AI.leage Dashboard Event Bus, Schemas, and HTTP Server."""

import json
import time
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from mileage.dashboard import (
    DashboardEvent,
    DashboardEventBus,
    DashboardStats,
    PipelineStage,
    PipelineStatus,
    BuildRecord,
)
from mileage.dashboard.server import (
    DashboardServer,
    get_local_ip,
)


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Ensure clean event bus singleton for every test."""
    DashboardEventBus.reset_instance()
    yield
    DashboardEventBus.reset_instance()


class TestDashboardEvent:
    def test_sse_formatting(self):
        event = DashboardEvent(
            event_type="test_event",
            data={"foo": "bar", "num": 42},
        )
        sse_str = event.to_sse()
        assert sse_str.startswith("data: ")
        assert sse_str.endswith("\n\n")

        # Extract JSON
        raw_json = sse_str[len("data: ") : -2]
        payload = json.loads(raw_json)
        assert payload["event_type"] == "test_event"
        assert payload["data"]["foo"] == "bar"
        assert payload["data"]["num"] == 42
        assert "timestamp" in payload


class TestDashboardEventBus:
    def test_singleton(self):
        bus1 = DashboardEventBus.get_instance()
        bus2 = DashboardEventBus.get_instance()
        assert bus1 is bus2

    def test_pipeline_update_publishing(self):
        bus = DashboardEventBus.get_instance()
        assert bus.pipeline_status.stage == PipelineStage.IDLE

        bus.publish_pipeline_update(
            stage=PipelineStage.BUILDING,
            goal="Test build",
            model_name="qwen2.5-coder",
            iteration=2,
            max_iterations=10,
            message="Editing file...",
        )

        assert bus.pipeline_status.stage == PipelineStage.BUILDING
        assert bus.pipeline_status.goal == "Test build"
        assert bus.pipeline_status.model_name == "qwen2.5-coder"
        assert bus.pipeline_status.iteration == 2
        assert bus.pipeline_status.message == "Editing file..."
        assert len(bus._event_log) == 1
        assert bus._event_log[0].event_type == "pipeline_update"

    def test_record_build_and_stats_recomputation(self):
        bus = DashboardEventBus.get_instance()

        build1 = BuildRecord(
            build_id="b1",
            goal="Goal 1",
            model_name="qwen2.5-coder",
            status="complete",
            started_at="2026-10-03T10:00:00Z",
            completed_at="2026-10-03T10:01:00Z",
            duration_ms=5000.0,
            tokens_used=10000,
            requirements_passed=5,
            requirements_total=5,
            escalated=False,
        )
        bus.record_build(build1)

        stats = bus.stats
        assert stats.projects_built == 1
        assert stats.total_tokens_used == 10000
        assert stats.tokens_saved_estimate == 10000
        assert stats.estimated_cost_avoided == 0.02  # 10k * 0.002 / 1000
        assert stats.avg_requirement_satisfaction == 100.0
        assert stats.tasks_completed_locally == 100.0
        assert "qwen2.5-coder" in stats.models_used
        assert stats.avg_build_time_ms == 5000.0

        # Record a second build that was escalated
        build2 = BuildRecord(
            build_id="b2",
            goal="Goal 2",
            model_name="gemma3:4b",
            status="escalated",
            started_at="2026-10-03T10:05:00Z",
            completed_at="2026-10-03T10:07:00Z",
            duration_ms=15000.0,
            tokens_used=20000,
            requirements_passed=1,
            requirements_total=2,
            escalated=True,
        )
        bus.record_build(build2)

        stats2 = bus.stats
        assert stats2.projects_built == 2
        assert stats2.total_tokens_used == 30000
        assert stats2.estimated_cost_avoided == 0.06
        # Average requirement satisfaction: (100 + 50) / 2 = 75.0
        assert stats2.avg_requirement_satisfaction == 75.0
        # 1 local out of 2 = 50.0%
        assert stats2.tasks_completed_locally == 50.0
        assert set(stats2.models_used) == {"qwen2.5-coder", "gemma3:4b"}
        assert stats2.avg_build_time_ms == 10000.0


class TestDashboardServer:
    def test_local_ip_detection(self):
        ip = get_local_ip()
        assert isinstance(ip, str)
        assert len(ip.split(".")) == 4

    def test_server_lifecycle_and_endpoints(self, tmp_path):
        # Create a mock static dir with index.html
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html><body>Dashboard</body></html>", encoding="utf-8")

        bus = DashboardEventBus.get_instance()
        bus.publish_pipeline_update(
            stage=PipelineStage.ROUTING,
            goal="Serve test",
            model_name="gemma3:4b",
        )

        server = DashboardServer(
            port=31415,
            host="127.0.0.1",
            static_dir=static_dir,
        )

        server.start(blocking=False)
        assert server.is_running
        time.sleep(0.5)

        try:
            import urllib.request

            # 1. Test GET /
            with urllib.request.urlopen("http://127.0.0.1:31415/") as resp:
                assert resp.status == 200
                assert b"Dashboard" in resp.read()

            # 2. Test GET /api/status
            with urllib.request.urlopen("http://127.0.0.1:31415/api/status") as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode())
                assert data["stage"] == "routing"
                assert data["goal"] == "Serve test"

            # 3. Test GET /api/stats
            with urllib.request.urlopen("http://127.0.0.1:31415/api/stats") as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode())
                assert "projects_built" in data
                assert "estimated_cost_avoided" in data

            # 4. Test GET /api/health
            with urllib.request.urlopen("http://127.0.0.1:31415/api/health") as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode())
                assert data["status"] == "ok"

        finally:
            server.stop()
            assert not server.is_running
