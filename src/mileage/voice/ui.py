"""Rich animated terminal interface for hands-free voice interaction."""

import time
from typing import Optional
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mileage.voice.states import STATE_DISPLAYS, VoiceState


class VoiceTerminalUI:
    """Renders a responsive, animated terminal UI for the hands-free loop."""

    ANIMATION_FRAMES = [
        "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
    ]
    WAVE_FRAMES = [
        " ▂▃▄▅▆▇█",
        "▂▃▄▅▆▇█ ",
        "▃▄▅▆▇█ ▂",
        "▄▅▆▇█ ▂▃",
        "▅▆▇█ ▂▃▄",
        "▆▇█ ▂▃▄▅",
        "▇█ ▂▃▄▅▆",
        "█ ▂▃▄▅▆▇",
    ]

    def __init__(self):
        self.state: VoiceState = VoiceState.IDLE
        self.user_transcript: str = ""
        self.assistant_response: str = ""
        self.spoken_sentence: str = ""
        self.status_message: str = "Ready. Speak freely or say 'stop' to pause."
        self.last_latency_ms: float = 0.0
        self.frame_index = 0

    def update_state(self, state: VoiceState, message: Optional[str] = None) -> None:
        """Update active state and optional status message."""
        self.state = state
        if message:
            self.status_message = message

    def set_transcript(self, text: str) -> None:
        """Set the latest transcribed user speech."""
        self.user_transcript = text

    def set_response(self, full_response: str, spoken_response: str, latency_ms: float = 0.0) -> None:
        """Set the assistant's generated response and spoken excerpt."""
        self.assistant_response = full_response
        self.spoken_sentence = spoken_response
        self.last_latency_ms = latency_ms

    def render(self) -> RenderableType:
        """Render the complete live UI frame."""
        self.frame_index = (self.frame_index + 1) % len(self.ANIMATION_FRAMES)
        anim_char = self.ANIMATION_FRAMES[self.frame_index]
        wave_str = self.WAVE_FRAMES[self.frame_index % len(self.WAVE_FRAMES)]

        disp = STATE_DISPLAYS.get(self.state, STATE_DISPLAYS[VoiceState.IDLE])

        # Header with State & Audio Visualizer
        state_text = Text()
        state_text.append(f" {anim_char} ", style="bold cyan")
        state_text.append(f"[{disp.label}] ", style=disp.color)
        state_text.append(f"{disp.description} ", style="white")

        if self.state == VoiceState.LISTENING:
            state_text.append(f"  {wave_str} ", style="bold green")
        elif self.state == VoiceState.SPEAKING:
            state_text.append(f"  {wave_str} [MIC MUTED] ", style="bold magenta")
        elif self.state == VoiceState.PROCESSING:
            state_text.append("  (thinking...) ", style="bold yellow")

        # Content Table
        grid = Table.grid(padding=(0, 1))
        grid.add_column("Key", style="bold dim white", width=14)
        grid.add_column("Value")

        # Status
        grid.add_row("Status:", Text(self.status_message, style="italic dim"))

        # User Speech
        user_styled = Text(self.user_transcript or "(listening for speech...)", style="cyan" if self.user_transcript else "dim")
        grid.add_row("You said:", user_styled)

        # Assistant Response
        if self.assistant_response:
            resp_styled = Text(self.assistant_response, style="green")
            grid.add_row("m.AI.leage:", resp_styled)

        # Spoken audio output
        if self.spoken_sentence:
            spoken_styled = Text(f'"{self.spoken_sentence}"', style="bold magenta")
            grid.add_row("Spoken (TTS):", spoken_styled)

        # Metrics / footer info
        footer = Text()
        if self.last_latency_ms > 0:
            footer.append(f"Latency: {self.last_latency_ms:.0f}ms  |  ", style="dim")
        footer.append("Say 'stop' to halt  |  Ctrl+C to exit", style="dim cyan")

        main_panel = Panel(
            grid,
            title=state_text,
            subtitle=footer,
            subtitle_align="right",
            border_style=disp.color.split()[-1],
            padding=(1, 2),
        )

        return main_panel
