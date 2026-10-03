"""Rich console and styling configuration for m.AI.leage."""

import sys
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text

# Ensure Windows terminal handles UTF-8 smoothly without charmap errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "danger": "bold red",
    "success": "bold green",
    "accent": "bold magenta",
    "muted": "dim white",
    "highlight": "bold cyan",
})

console = Console(theme=custom_theme)
err_console = Console(theme=custom_theme, stderr=True)


def print_banner() -> None:
    """Print the m.AI.leage visual header."""
    banner_text = Text()
    banner_text.append("m", style="bold cyan")
    banner_text.append(".", style="bold white")
    banner_text.append("AI", style="bold magenta")
    banner_text.append(".", style="bold white")
    banner_text.append("leage", style="bold cyan")
    banner_text.append("  |  Local-First AI CLI & Workspace Telemetry", style="dim white")

    panel = Panel(
        banner_text,
        border_style="magenta",
        subtitle="[dim]Zero Cloud - 100% Local - Ollama Powered[/dim]",
        subtitle_align="right",
    )
    console.print(panel)


def print_speedometer_greeting(
    model_name: str = "local",
    voice_active: bool = True,
    workspace_name: str = "",
) -> None:
    """Render the clean, minimal m.AI.leage Claude-Code-like speedometer mascot greeting."""
    from rich.table import Table
    from mileage.ui.mascot import get_app_mascot

    art = get_app_mascot()

    info = Text()
    info.append("m", style="bold cyan")
    info.append(".", style="bold white")
    info.append("AI", style="bold magenta")
    info.append(".", style="bold white")
    info.append("leage", style="bold cyan")
    info.append("  Autonomous Local AI\n", style="bold white")
    info.append("Local Autonomous Agent Orchestration\n\n", style="dim")

    info.append("  ● ", style="bold green")
    info.append("100% Local", style="bold white")
    info.append(" • Zero Cloud Telemetry\n", style="dim")

    info.append("  ● ", style="bold cyan")
    info.append("Model: ", style="dim")
    info.append(f"{model_name}\n", style="bold cyan")

    info.append("  ● ", style="bold magenta" if voice_active else "dim")
    info.append("Voice: ", style="dim")
    info.append(
        "Whisper + VAD + TTS (Hands-Free)\n" if voice_active else "Text Mode Only\n",
        style="bold magenta" if voice_active else "dim",
    )

    if workspace_name:
        info.append("  ● ", style="bold yellow")
        info.append("Workspace: ", style="dim")
        info.append(f"{workspace_name} (.mileage/)\n", style="white")

    grid = Table.grid(padding=(0, 3))
    grid.add_row(art, info)

    panel = Panel(
        grid,
        border_style="cyan",
        padding=(1, 2),
        subtitle="[dim]Speak or type your goal • 'exit' or Ctrl+C to quit • 'mileage serve' for web UI[/dim]",
        subtitle_align="center",
    )
    console.print(panel)
    console.print()
