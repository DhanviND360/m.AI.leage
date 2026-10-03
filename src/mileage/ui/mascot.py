"""Terminal mascot logo renderer for m.AI.leage.

Renders the cute speedometer companion mascot in the terminal,
matching the application icon (colored gauge arc, needle, smiling face,
thumbs-up gesture, and feet).
"""

import sys
from pathlib import Path
from typing import Optional
from rich.text import Text

# Safe UTF-8 reconfiguration for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_mascot_text_art() -> Text:
    """Return the stylized Rich Text art of the m.AI.leage mascot.
    
    Faithfully depicts the official icon:
    - Multi-color gauge arc: Green (#22c55e) -> Yellow (#eab308) -> Orange (#f97316) -> Red (#ef4444)
    - Pivot dot and white needle pointing up-right (↗)
    - Cute curved eyes ( ^ ‿ ^ ) with happy open smile
    - Thumbs up arm (👍) on the right
    - Cute shoe feet at the bottom
    """
    art = Text()

    # Line 1: Top dome outline
    art.append("         .─────────.         \n", style="bold white")

    # Line 2: Upper dial arc (Green, Yellow, Orange, Red)
    art.append("      .─' ", style="bold white")
    art.append("█", style="bold #22c55e")
    art.append(" ", style="default")
    art.append("█", style="bold #eab308")
    art.append(" ", style="default")
    art.append("█", style="bold #f97316")
    art.append(" ", style="default")
    art.append("█", style="bold #ef4444")
    art.append(" '─.        \n", style="bold white")

    # Line 3: Dial needle pointing up-right
    art.append("    .─' ", style="bold white")
    art.append("█", style="bold #22c55e")
    art.append("   ", style="default")
    art.append("●", style="bold white")
    art.append("──", style="bold white")
    art.append("↗", style="bold white")
    art.append("  ", style="default")
    art.append("█", style="bold #ef4444")
    art.append(" '─.      \n", style="bold white")

    # Line 4: Thumbs up gesture and side body
    art.append("   │ ", style="bold white")
    art.append("█", style="bold #22c55e")
    art.append("    /       ", style="default")
    art.append("█", style="bold #ef4444")
    art.append(" │─(b )   \n", style="bold white")

    # Line 5: Happy face eyes
    art.append("   │   ( ^  ‿  ^ )  │  👍     \n", style="bold white")

    # Line 6: Happy mouth / smile
    art.append("    \\      ", style="bold white")
    art.append("( ᗨ )", style="bold #f43f5e")
    art.append("   /        \n", style="bold white")

    # Line 7: Bottom curve
    art.append("     '─._________.─'         \n", style="bold white")

    # Line 8 & 9: Cute little mascot feet
    art.append("        / \\   / \\           \n", style="bold white")
    art.append("       (___) (___)          \n", style="bold white")

    return art


def get_image_pixel_art(
    image_path: Optional[Path] = None,
    width: int = 24,
    height: int = 20,
) -> Optional[Text]:
    """Optionally render the PNG directly using PIL RGB half-block sampling."""
    try:
        from PIL import Image
        from rich.style import Style
        from rich.color import Color

        if image_path is None:
            image_path = Path(__file__).parent.parent / "assets" / "speedy_mascot.png"

        if not image_path.is_file():
            return None

        img = Image.open(image_path).convert("RGBA")
        resized = img.resize((width, height), Image.Resampling.LANCZOS)

        art = Text()
        for y in range(0, height, 2):
            for x in range(width):
                r1, g1, b1, a1 = resized.getpixel((x, y))
                r2, g2, b2, a2 = resized.getpixel((x, y + 1)) if y + 1 < height else (0, 0, 0, 0)

                is_bg1 = a1 < 40 or (r1 < 30 and g1 < 30 and b1 < 30)
                is_bg2 = a2 < 40 or (r2 < 30 and g2 < 30 and b2 < 30)

                if is_bg1 and is_bg2:
                    art.append(" ")
                elif is_bg2:
                    art.append("▀", style=Style(color=Color.from_rgb(r1, g1, b1)))
                elif is_bg1:
                    art.append("▄", style=Style(color=Color.from_rgb(r2, g2, b2)))
                else:
                    art.append("▀", style=Style(color=Color.from_rgb(r1, g1, b1), bgcolor=Color.from_rgb(r2, g2, b2)))
            art.append("\n")

        return art
    except Exception:
        return None


def get_app_mascot(prefer_pixel_art: bool = False) -> Text:
    """Return the mascot art for the terminal interface."""
    if prefer_pixel_art:
        pixel_art = get_image_pixel_art()
        if pixel_art:
            return pixel_art
    return get_mascot_text_art()
