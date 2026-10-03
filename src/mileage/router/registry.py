"""Capability registry for local Ollama models.

Maps model families and names to capability scores using heuristic rules.
Scores range from 0.0 (no ability) to 1.0 (exceptional).
These are starting points — benchmarks refine them over time.
"""

from typing import Dict, List, Optional, Set

from mileage.router.schemas import (
    Capability,
    CapabilityScore,
    ModelProfile,
)


# ── Heuristic Capability Profiles ────────────────────────────────────────────
# Each entry maps a model family/name pattern to a dict of capability scores.
# Scores: 0.0 = cannot, 0.3 = basic, 0.5 = adequate, 0.7 = good, 0.9 = excellent

_FAMILY_CAPABILITIES: Dict[str, Dict[Capability, float]] = {
    # ── Gemma family ──
    "gemma": {
        Capability.CODING: 0.6,
        Capability.REASONING: 0.6,
        Capability.CHAT: 0.7,
        Capability.INSTRUCTION: 0.7,
        Capability.SUMMARIZATION: 0.6,
        Capability.PLANNING: 0.5,
        Capability.SPEED: 0.7,
    },
    "gemma2": {
        Capability.CODING: 0.7,
        Capability.REASONING: 0.7,
        Capability.CHAT: 0.8,
        Capability.INSTRUCTION: 0.8,
        Capability.SUMMARIZATION: 0.7,
        Capability.PLANNING: 0.6,
        Capability.SPEED: 0.6,
    },
    "gemma3": {
        Capability.CODING: 0.8,
        Capability.REASONING: 0.8,
        Capability.VISION: 0.7,
        Capability.CHAT: 0.8,
        Capability.INSTRUCTION: 0.8,
        Capability.SUMMARIZATION: 0.7,
        Capability.PLANNING: 0.7,
        Capability.SPEED: 0.6,
    },

    # ── Llama family ──
    "llama": {
        Capability.CODING: 0.5,
        Capability.REASONING: 0.5,
        Capability.CHAT: 0.6,
        Capability.INSTRUCTION: 0.6,
        Capability.SUMMARIZATION: 0.5,
        Capability.SPEED: 0.6,
    },
    "llama3": {
        Capability.CODING: 0.7,
        Capability.REASONING: 0.7,
        Capability.CHAT: 0.8,
        Capability.INSTRUCTION: 0.8,
        Capability.SUMMARIZATION: 0.7,
        Capability.PLANNING: 0.6,
        Capability.SPEED: 0.5,
    },
    "codellama": {
        Capability.CODING: 0.9,
        Capability.REASONING: 0.5,
        Capability.CHAT: 0.4,
        Capability.INSTRUCTION: 0.6,
        Capability.SPEED: 0.6,
    },

    # ── Mistral family ──
    "mistral": {
        Capability.CODING: 0.7,
        Capability.REASONING: 0.7,
        Capability.CHAT: 0.7,
        Capability.INSTRUCTION: 0.7,
        Capability.SUMMARIZATION: 0.6,
        Capability.PLANNING: 0.6,
        Capability.SPEED: 0.5,
    },
    "mixtral": {
        Capability.CODING: 0.8,
        Capability.REASONING: 0.8,
        Capability.CHAT: 0.8,
        Capability.INSTRUCTION: 0.8,
        Capability.SUMMARIZATION: 0.7,
        Capability.PLANNING: 0.7,
        Capability.SPEED: 0.3,  # Large MoE, slower
    },

    # ── Phi family ──
    "phi": {
        Capability.CODING: 0.5,
        Capability.REASONING: 0.5,
        Capability.CHAT: 0.5,
        Capability.SPEED: 0.9,
    },
    "phi3": {
        Capability.CODING: 0.7,
        Capability.REASONING: 0.7,
        Capability.CHAT: 0.7,
        Capability.INSTRUCTION: 0.7,
        Capability.SPEED: 0.8,
    },
    "phi4": {
        Capability.CODING: 0.8,
        Capability.REASONING: 0.8,
        Capability.CHAT: 0.8,
        Capability.INSTRUCTION: 0.8,
        Capability.PLANNING: 0.6,
        Capability.SPEED: 0.6,
    },

    # ── Qwen family ──
    "qwen": {
        Capability.CODING: 0.7,
        Capability.REASONING: 0.7,
        Capability.CHAT: 0.7,
        Capability.INSTRUCTION: 0.7,
        Capability.SPEED: 0.5,
    },
    "qwen2": {
        Capability.CODING: 0.8,
        Capability.REASONING: 0.8,
        Capability.CHAT: 0.8,
        Capability.INSTRUCTION: 0.8,
        Capability.SUMMARIZATION: 0.7,
        Capability.PLANNING: 0.7,
        Capability.SPEED: 0.5,
    },
    "qwen2.5": {
        Capability.CODING: 0.85,
        Capability.REASONING: 0.85,
        Capability.CHAT: 0.85,
        Capability.INSTRUCTION: 0.85,
        Capability.SUMMARIZATION: 0.8,
        Capability.PLANNING: 0.8,
        Capability.SPEED: 0.5,
    },
    "qwen2.5-coder": {
        Capability.CODING: 0.9,
        Capability.REASONING: 0.7,
        Capability.CHAT: 0.6,
        Capability.INSTRUCTION: 0.8,
        Capability.SPEED: 0.5,
    },

    # ── DeepSeek family ──
    "deepseek": {
        Capability.CODING: 0.7,
        Capability.REASONING: 0.7,
        Capability.CHAT: 0.6,
        Capability.SPEED: 0.5,
    },
    "deepseek-coder": {
        Capability.CODING: 0.9,
        Capability.REASONING: 0.6,
        Capability.INSTRUCTION: 0.7,
        Capability.SPEED: 0.5,
    },
    "deepseek-r1": {
        Capability.CODING: 0.8,
        Capability.REASONING: 0.95,
        Capability.PLANNING: 0.8,
        Capability.INSTRUCTION: 0.8,
        Capability.SPEED: 0.3,
    },

    # ── StarCoder family ──
    "starcoder": {
        Capability.CODING: 0.85,
        Capability.REASONING: 0.4,
        Capability.CHAT: 0.3,
        Capability.SPEED: 0.6,
    },
    "starcoder2": {
        Capability.CODING: 0.9,
        Capability.REASONING: 0.5,
        Capability.CHAT: 0.4,
        Capability.INSTRUCTION: 0.6,
        Capability.SPEED: 0.6,
    },

    # ── CodeGemma ──
    "codegemma": {
        Capability.CODING: 0.85,
        Capability.REASONING: 0.5,
        Capability.CHAT: 0.4,
        Capability.INSTRUCTION: 0.6,
        Capability.SPEED: 0.7,
    },

    # ── TinyLlama ──
    "tinyllama": {
        Capability.CHAT: 0.4,
        Capability.SPEED: 0.95,
        Capability.SUMMARIZATION: 0.3,
    },

    # ── Yi ──
    "yi": {
        Capability.CODING: 0.6,
        Capability.REASONING: 0.7,
        Capability.CHAT: 0.7,
        Capability.INSTRUCTION: 0.7,
        Capability.SPEED: 0.5,
    },

    # ── Command-R ──
    "command-r": {
        Capability.CODING: 0.7,
        Capability.REASONING: 0.8,
        Capability.CHAT: 0.8,
        Capability.INSTRUCTION: 0.8,
        Capability.SUMMARIZATION: 0.8,
        Capability.PLANNING: 0.7,
        Capability.SPEED: 0.3,
    },
}

# ── Name-based overrides (take precedence over family) ──
_NAME_PATTERN_CAPABILITIES: Dict[str, Dict[Capability, float]] = {
    "llava": {Capability.VISION: 0.8, Capability.CHAT: 0.6},
    "bakllava": {Capability.VISION: 0.7, Capability.CHAT: 0.5},
    "moondream": {Capability.VISION: 0.6, Capability.CHAT: 0.4, Capability.SPEED: 0.8},
}


def _match_family(family: str, name: str) -> Optional[Dict[Capability, float]]:
    """Find the best matching capability profile for a model."""
    family_lower = family.lower().strip()
    name_lower = name.lower().split(":")[0]

    # 1. Check name-based patterns first (vision models, etc.)
    for pattern, caps in _NAME_PATTERN_CAPABILITIES.items():
        if pattern in name_lower:
            return caps

    # 2. Exact family match
    if family_lower in _FAMILY_CAPABILITIES:
        return _FAMILY_CAPABILITIES[family_lower]

    # 3. Name-based family inference
    for fam_key in sorted(_FAMILY_CAPABILITIES.keys(), key=len, reverse=True):
        if name_lower.startswith(fam_key):
            return _FAMILY_CAPABILITIES[fam_key]

    return None


def _adjust_for_size(
    caps: Dict[Capability, float],
    parameter_billions: Optional[float],
) -> Dict[Capability, float]:
    """Adjust capability scores based on model size.

    Smaller models get a speed bonus but quality penalty.
    Larger models get a quality bonus but speed penalty.
    """
    if parameter_billions is None:
        return caps

    adjusted = dict(caps)
    b = parameter_billions

    if b <= 1.0:
        # Very small — fast but limited
        adjusted[Capability.SPEED] = min(1.0, adjusted.get(Capability.SPEED, 0.5) + 0.3)
        for cap in [Capability.CODING, Capability.REASONING, Capability.PLANNING]:
            if cap in adjusted:
                adjusted[cap] = max(0.1, adjusted[cap] - 0.2)
    elif b <= 4.0:
        # Small — good speed
        adjusted[Capability.SPEED] = min(1.0, adjusted.get(Capability.SPEED, 0.5) + 0.15)
    elif b >= 30.0:
        # Large — slower but more capable
        adjusted[Capability.SPEED] = max(0.1, adjusted.get(Capability.SPEED, 0.5) - 0.2)
        for cap in [Capability.CODING, Capability.REASONING, Capability.PLANNING]:
            if cap in adjusted:
                adjusted[cap] = min(1.0, adjusted[cap] + 0.1)
    elif b >= 70.0:
        adjusted[Capability.SPEED] = max(0.05, adjusted.get(Capability.SPEED, 0.5) - 0.3)

    return adjusted


def populate_capabilities(profile: ModelProfile) -> ModelProfile:
    """Populate a ModelProfile with heuristic capability scores.

    Looks up the model family/name in the registry, adjusts for
    parameter size, and attaches scored capabilities.

    Returns the same profile object, mutated in-place.
    """
    base_caps = _match_family(profile.family, profile.name)

    if base_caps is None:
        # Unknown model — assign conservative defaults
        base_caps = {
            Capability.CHAT: 0.5,
            Capability.SPEED: 0.5,
        }

    # Adjust for model size
    adjusted = _adjust_for_size(base_caps, profile.parameter_count_billions)

    # Build CapabilityScore list
    profile.capabilities = [
        CapabilityScore(
            capability=cap,
            score=round(score, 2),
            source="heuristic",
        )
        for cap, score in sorted(adjusted.items(), key=lambda x: -x[1])
    ]

    return profile


def populate_all(profiles: List[ModelProfile]) -> List[ModelProfile]:
    """Populate capabilities for all discovered model profiles."""
    for p in profiles:
        populate_capabilities(p)
    return profiles
