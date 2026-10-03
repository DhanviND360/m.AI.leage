"""UI subsystem for m.AI.leage."""

from mileage.ui.console import console, err_console, print_banner
from mileage.ui.components import (
    render_error,
    render_success,
    render_models_table,
    render_workspace_table,
    render_metrics_table,
)
from mileage.ui.doctor_view import DoctorDiagnostics, render_doctor_report

__all__ = [
    "console",
    "err_console",
    "print_banner",
    "render_error",
    "render_success",
    "render_models_table",
    "render_workspace_table",
    "render_metrics_table",
    "DoctorDiagnostics",
    "render_doctor_report",
]
