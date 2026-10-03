"""Voice state machine enumeration and state metadata."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class VoiceState(str, Enum):
    """Execution states for hands-free voice interaction."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    BUILDING = "BUILDING"
    COMPLETE = "COMPLETE"


@dataclass
class StateDisplay:
    """Rich display properties for voice states."""

    label: str
    color: str
    symbol: str
    description: str


STATE_DISPLAYS = {
    VoiceState.IDLE: StateDisplay(
        label="IDLE",
        color="dim cyan",
        symbol="[dim]o[/dim]",
        description="Waiting for speech...",
    ),
    VoiceState.LISTENING: StateDisplay(
        label="LISTENING",
        color="bold green",
        symbol="[bold green]●[/bold green]",
        description="Listening to user voice input...",
    ),
    VoiceState.PROCESSING: StateDisplay(
        label="PROCESSING",
        color="bold yellow",
        symbol="[bold yellow]⚙[/bold yellow]",
        description="Transcribing and thinking...",
    ),
    VoiceState.SPEAKING: StateDisplay(
        label="SPEAKING",
        color="bold magenta",
        symbol="[bold magenta]🔊[/bold magenta]",
        description="Speaking response (mic muted)...",
    ),
    VoiceState.BUILDING: StateDisplay(
        label="BUILDING",
        color="bold blue",
        symbol="[bold blue]🔨[/bold blue]",
        description="Executing workspace action...",
    ),
    VoiceState.COMPLETE: StateDisplay(
        label="COMPLETE",
        color="bold green",
        symbol="[bold green]✓[/bold green]",
        description="Turn completed.",
    ),
}
