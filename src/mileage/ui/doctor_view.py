"""Doctor diagnostics engine and Rich rendering view."""

import sys
import platform
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mileage.core.config import MileageConfig
from mileage.core.workspace import WorkspaceManager
from mileage.metrics.tracker import MetricsTracker
from mileage.models.ollama_client import OllamaClient
from mileage.models.schemas import (
    DoctorCheckItem,
    DoctorCheckStatus,
    DoctorReport,
)
from mileage.ui.console import console


class DoctorDiagnostics:
    """Performs deep health and system checks for m.AI.leage."""

    def __init__(self, workspace: Optional[WorkspaceManager] = None):
        self.workspace = workspace or WorkspaceManager.find_workspace() or WorkspaceManager()
        self.config = self.workspace.get_config()
        self.ollama = OllamaClient(
            host=self.config.ollama.host,
            timeout_seconds=self.config.ollama.timeout_seconds,
        )

    def run_all_checks(self) -> DoctorReport:
        """Run all diagnostic checks and assemble a DoctorReport."""
        report = DoctorReport(
            python_version=sys.version.split()[0],
            os_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
        )

        # 1. Environment & Runtime Check
        self._check_runtime(report)

        # 2. Workspace Status Check
        self._check_workspace(report)

        # 3. Ollama Service Connectivity
        self._check_ollama_service(report)

        # 4. Verified Local Models
        self._check_installed_models(report)

        # 5. Local Metrics Store
        self._check_metrics_store(report)

        # 6. Hands-Free Voice Stack
        self._check_voice_stack(report)

        return report

    def _check_runtime(self, report: DoctorReport) -> None:
        """Verify Python version and runtime environment."""
        py_major, py_minor = sys.version_info[:2]
        if py_major == 3 and py_minor in (11, 12):
            report.checks.append(
                DoctorCheckItem(
                    category="Runtime",
                    name="Python Version",
                    status=DoctorCheckStatus.OK,
                    message=f"Python {py_major}.{py_minor} supported ({sys.version.split()[0]})",
                )
            )
        elif py_major == 3 and py_minor >= 10:
            report.checks.append(
                DoctorCheckItem(
                    category="Runtime",
                    name="Python Version",
                    status=DoctorCheckStatus.WARNING,
                    message=f"Python {py_major}.{py_minor} detected; Python 3.12 recommended.",
                )
            )
        else:
            report.checks.append(
                DoctorCheckItem(
                    category="Runtime",
                    name="Python Version",
                    status=DoctorCheckStatus.ERROR,
                    message=f"Unsupported Python version: {sys.version.split()[0]}",
                    fix_suggestion="Upgrade to Python 3.11 or 3.12.",
                )
            )

    def _check_workspace(self, report: DoctorReport) -> None:
        """Verify workspace initialization and configuration."""
        if self.workspace.is_initialized():
            stats = self.workspace.get_stats()
            report.checks.append(
                DoctorCheckItem(
                    category="Workspace",
                    name="Initialization",
                    status=DoctorCheckStatus.OK,
                    message=f"Workspace initialized ({stats.project_name})",
                    details=f"Root: {self.workspace.root_dir}\nFiles tracked: {stats.total_files:,}",
                )
            )
        else:
            report.checks.append(
                DoctorCheckItem(
                    category="Workspace",
                    name="Initialization",
                    status=DoctorCheckStatus.WARNING,
                    message="Directory not initialized as m.AI.leage workspace.",
                    fix_suggestion="Run 'mileage init' to configure this project.",
                )
            )

    def _check_ollama_service(self, report: DoctorReport) -> None:
        """Verify Ollama daemon connection, auto-starting if stopped."""
        # Attempt auto-launch if not currently running
        self.ollama.start_daemon(timeout_seconds=4.0)
        health = self.ollama.check_health(quick_timeout=3.0)

        if health.is_running:
            report.checks.append(
                DoctorCheckItem(
                    category="Ollama",
                    name="Daemon Connectivity",
                    status=DoctorCheckStatus.OK,
                    message=f"Connected to local Ollama at {health.host} (auto-managed)",
                    details=f"Server version: {health.version} | Latency: {health.response_time_ms}ms",
                )
            )
        else:
            report.checks.append(
                DoctorCheckItem(
                    category="Ollama",
                    name="Daemon Connectivity",
                    status=DoctorCheckStatus.ERROR,
                    message=f"Cannot connect to local Ollama at {self.config.ollama.host}",
                    details=health.error_message,
                    fix_suggestion="Start Ollama locally by running 'ollama serve' in another terminal or open the Ollama app.",
                )
            )

    def _check_installed_models(self, report: DoctorReport) -> None:
        """Verify installed models automatically and auto-adopt installed model."""
        health = self.ollama.check_health(quick_timeout=2.0)
        if not health.is_running:
            report.checks.append(
                DoctorCheckItem(
                    category="Models",
                    name="Verified Models",
                    status=DoctorCheckStatus.WARNING,
                    message="Skipped model check: Ollama daemon not reachable.",
                    fix_suggestion="Start Ollama and re-run 'mileage doctor'.",
                )
            )
            return

        try:
            models = self.ollama.list_models()
            default_mod = self.config.ollama.default_model

            if not models:
                report.checks.append(
                    DoctorCheckItem(
                        category="Models",
                        name="Verified Models",
                        status=DoctorCheckStatus.WARNING,
                        message="0 local models found in Ollama.",
                        fix_suggestion="Pull a fast local model, e.g.: 'ollama pull llama3.2' or 'ollama pull mistral'",
                    )
                )
                return

            has_default = any(
                m.name.lower() == default_mod.lower()
                or m.name.split(":")[0] == default_mod.split(":")[0]
                for m in models
            )

            model_names = [m.name for m in models]
            detail_str = f"Installed models ({len(models)}): {', '.join(model_names)}"

            if has_default:
                report.checks.append(
                    DoctorCheckItem(
                        category="Models",
                        name="Verified Models",
                        status=DoctorCheckStatus.OK,
                        message=f"{len(models)} models verified; default '{default_mod}' is ready.",
                        details=detail_str,
                    )
                )
            else:
                # Automatically adopt the first installed model!
                adopted_model = models[0].name
                self.config.ollama.default_model = adopted_model
                if self.workspace.is_initialized():
                    self.config.save_to_dir(self.workspace.root_dir)

                report.checks.append(
                    DoctorCheckItem(
                        category="Models",
                        name="Verified Models",
                        status=DoctorCheckStatus.OK,
                        message=f"{len(models)} model(s) verified; automatically selected active '{adopted_model}'.",
                        details=detail_str,
                    )
                )
        except Exception as e:
            report.checks.append(
                DoctorCheckItem(
                    category="Models",
                    name="Verified Models",
                    status=DoctorCheckStatus.ERROR,
                    message=f"Error inspecting local models: {str(e)}",
                )
            )

    def _check_metrics_store(self, report: DoctorReport) -> None:
        """Verify local metrics log storage."""
        if not self.workspace.is_initialized():
            report.checks.append(
                DoctorCheckItem(
                    category="Metrics",
                    name="Telemetry Store",
                    status=DoctorCheckStatus.WARNING,
                    message="Metrics store not initialized (workspace uninitialized).",
                )
            )
            return

        metrics_file = self.workspace.metrics_path
        tracker = MetricsTracker(metrics_file)
        summary = tracker.get_summary()

        report.checks.append(
            DoctorCheckItem(
                category="Metrics",
                name="Telemetry Store",
                status=DoctorCheckStatus.OK,
                message=f"Local store active ({summary.total_runs} recorded runs)",
                details=f"Path: {metrics_file}\nTotal tokens processed: {summary.total_tokens:,}",
            )
        )

    def _check_voice_stack(self, report: DoctorReport) -> None:
        """Verify local voice stack (faster-whisper, pyttsx3, audio subsystem)."""
        stt_ok = False
        tts_ok = False
        audio_ok = False

        try:
            import faster_whisper
            stt_ok = True
        except ImportError:
            pass

        try:
            import pyttsx3
            tts_ok = True
        except ImportError:
            pass

        try:
            import sounddevice as sd
            devices = sd.query_devices()
            audio_ok = len(devices) > 0
        except Exception:
            pass

        if stt_ok and tts_ok:
            status = DoctorCheckStatus.OK
            msg = "Local STT (faster-whisper) & TTS (pyttsx3) ready."
            details = f"Audio subsystem: {'detected' if audio_ok else 'audio hardware query pending'}"
            sugg = None
        else:
            status = DoctorCheckStatus.WARNING
            missing = []
            if not stt_ok:
                missing.append("faster-whisper")
            if not tts_ok:
                missing.append("pyttsx3")
            msg = f"Voice stack incomplete. Missing: {', '.join(missing)}"
            details = None
            sugg = f"Run 'pip install {' '.join(missing)}' to enable voice."

        report.checks.append(
            DoctorCheckItem(
                category="Voice",
                name="Hands-Free Stack",
                status=status,
                message=msg,
                details=details,
                fix_suggestion=sugg,
            )
        )


def render_doctor_report(report: DoctorReport) -> None:
    """Render the DoctorReport into a clean, modern Rich view."""
    table = Table(
        title="[bold magenta]m.AI.leage Doctor Health Report[/bold magenta]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Category", style="bold white", width=12)
    table.add_column("Check", style="white", width=22)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Result & Details", style="dim white")

    status_badges = {
        DoctorCheckStatus.OK: "[bold green]PASS[/bold green]",
        DoctorCheckStatus.WARNING: "[bold yellow]WARN[/bold yellow]",
        DoctorCheckStatus.ERROR: "[bold red]FAIL[/bold red]",
    }

    for item in report.checks:
        badge = status_badges.get(item.status, "[dim]UNK[/dim]")
        content = item.message
        if item.details:
            content += f"\n[dim]{item.details}[/dim]"
        if item.fix_suggestion:
            content += f"\n[yellow][!] Suggestion: {item.fix_suggestion}[/yellow]"

        table.add_row(
            item.category,
            item.name,
            badge,
            content,
        )

    console.print(table)

    # Summary Panel
    summary_text = Text()
    summary_text.append(f"OS: {report.os_info}  |  Python: {report.python_version}\n", style="dim white")
    summary_text.append(f"Passed: {report.ok_count}   ", style="bold green")
    summary_text.append(f"Warnings: {report.warning_count}   ", style="bold yellow")
    summary_text.append(f"Errors: {report.error_count}", style="bold red")

    border = "green" if report.is_healthy else ("yellow" if report.error_count == 0 else "red")
    title_status = "All Systems Ready" if report.is_healthy else ("Ready with Warnings" if report.error_count == 0 else "Action Required")

    console.print(
        Panel(
            summary_text,
            title=f"[bold {border}]{title_status}[/bold {border}]",
            border_style=border,
            padding=(0, 2),
        )
    )
