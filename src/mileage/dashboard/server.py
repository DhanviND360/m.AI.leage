"""Dashboard HTTP server with REST API and SSE streaming.

Serves the React dashboard static files and provides:
- GET /api/status     → Current pipeline status
- GET /api/stats      → Aggregate statistics
- GET /api/history    → Build history
- GET /api/metrics    → Raw metrics from workspace
- GET /api/events     → SSE stream for live updates
- GET /              → React dashboard (index.html)

Binds to 0.0.0.0 so phones on the same Wi-Fi can connect.
"""

import asyncio
import json
import mimetypes
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

from mileage.core.logger import logger
from mileage.dashboard import (
    DashboardEvent,
    DashboardEventBus,
    PipelineStage,
)
from mileage.metrics.tracker import MetricsTracker
from mileage.ui.console import console


# ── Lightweight HTTP server using only stdlib ────────────────────────────
# We avoid heavy deps (FastAPI/Starlette) to keep the tool lightweight.
# Uses http.server with async SSE via threading.

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


def get_local_ip() -> str:
    """Detect the machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the m.AI.leage dashboard."""

    # Class-level references set by the server factory
    event_bus: DashboardEventBus
    static_dir: Path
    metrics_tracker: Optional[MetricsTracker] = None

    def log_message(self, format, *args):
        """Suppress default access logs to keep terminal clean."""
        pass

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # API routes
        if path == "/api/status":
            self._handle_status()
        elif path == "/api/stats":
            self._handle_stats()
        elif path == "/api/history":
            self._handle_history()
        elif path == "/api/metrics":
            self._handle_metrics()
        elif path == "/api/events":
            self._handle_sse()
        elif path == "/api/health":
            self._json_response({"status": "ok", "clients": self.event_bus.subscriber_count})
        else:
            self._handle_static(parsed.path)

    def _handle_status(self):
        self._json_response(self.event_bus.pipeline_status.model_dump())

    def _handle_stats(self):
        self._json_response(self.event_bus.stats.model_dump())

    def _handle_history(self):
        builds = [b.model_dump() for b in self.event_bus.build_history]
        self._json_response(builds)

    def _handle_metrics(self):
        if self.metrics_tracker:
            summary = self.metrics_tracker.get_summary()
            self._json_response(summary.model_dump())
        else:
            self._json_response({})

    def _handle_sse(self):
        """Server-Sent Events stream for live dashboard updates."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._set_cors_headers()
        self.end_headers()

        # Send initial status
        initial = DashboardEvent(
            event_type="connected",
            data={
                "pipeline": self.event_bus.pipeline_status.model_dump(),
                "stats": self.event_bus.stats.model_dump(),
            },
        )
        try:
            self.wfile.write(initial.to_sse().encode("utf-8"))
            self.wfile.flush()
        except Exception:
            return

        # Create a sync-compatible polling mechanism
        # We use the event bus's event log and poll for new events
        last_seen = len(self.event_bus._event_log)
        try:
            while True:
                current_len = len(self.event_bus._event_log)
                if current_len > last_seen:
                    events = list(self.event_bus._event_log)
                    for ev in events[last_seen:]:
                        try:
                            self.wfile.write(ev.to_sse().encode("utf-8"))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            return
                    last_seen = current_len

                # Send keepalive every 15s
                try:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

                time.sleep(0.5)
        except Exception:
            pass

    def _handle_static(self, path: str):
        """Serve React dashboard static files."""
        if path == "/" or path == "":
            path = "/index.html"

        file_path = self.static_dir / path.lstrip("/")

        # SPA fallback: serve index.html for non-file routes
        if not file_path.is_file():
            file_path = self.static_dir / "index.html"

        if not file_path.is_file():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"

        try:
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            self.send_response(500)
            self.end_headers()


class DashboardServer:
    """Manages the m.AI.leage dashboard HTTP server lifecycle."""

    def __init__(
        self,
        port: int = 3000,
        host: str = "0.0.0.0",
        static_dir: Optional[Path] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
    ):
        self.port = port
        self.host = host
        self.static_dir = static_dir or (
            Path(__file__).parent / "static"
        )
        self.metrics_tracker = metrics_tracker
        self.event_bus = DashboardEventBus.get_instance()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = True) -> None:
        """Start the dashboard server."""
        # Configure handler class
        handler = type(
            "ConfiguredHandler",
            (DashboardRequestHandler,),
            {
                "event_bus": self.event_bus,
                "static_dir": self.static_dir,
                "metrics_tracker": self.metrics_tracker,
            },
        )

        self._server = HTTPServer((self.host, self.port), handler)
        self._server.timeout = 0.5

        local_ip = get_local_ip()

        logger.info(
            "Dashboard server starting on %s:%d", self.host, self.port
        )
        console.print(f"\n  [bold cyan]🚀 m.AI.leage Command Center[/bold cyan]")
        console.print(f"  [dim]────────────────────────────────[/dim]")
        console.print(f"  [white]Local:[/white]   [bold green]http://localhost:{self.port}[/bold green]")
        console.print(f"  [white]Network:[/white] [bold green]http://{local_ip}:{self.port}[/bold green]")
        console.print(f"  [dim]────────────────────────────────[/dim]")
        console.print(f"  [dim]Open on your phone to monitor builds![/dim]\n")

        if blocking:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            logger.info("Dashboard server stopped")

    @property
    def is_running(self) -> bool:
        return self._server is not None
