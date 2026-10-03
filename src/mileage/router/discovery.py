"""Auto-discover installed Ollama models with full metadata.

Probes each model for name, size, context length, family, quantization,
and the local hardware/runtime environment.
"""

import os
import platform
import sys
from typing import List, Optional

from mileage.core.logger import logger
from mileage.models.ollama_client import OllamaClient
from mileage.router.schemas import HardwareInfo, ModelProfile


# ── Known context lengths per family (conservative defaults) ──────────────
# Ollama doesn't always expose context length via API, so we maintain
# a lookup for well-known model families.
KNOWN_CONTEXT_LENGTHS = {
    "gemma": 8192,
    "gemma2": 8192,
    "gemma3": 8192,
    "llama": 4096,
    "llama2": 4096,
    "llama3": 8192,
    "llama3.1": 131072,
    "llama3.2": 131072,
    "codellama": 16384,
    "mistral": 32768,
    "mixtral": 32768,
    "phi": 2048,
    "phi3": 4096,
    "phi4": 16384,
    "qwen": 32768,
    "qwen2": 32768,
    "qwen2.5": 131072,
    "deepseek": 16384,
    "deepseek-coder": 16384,
    "deepseek-r1": 65536,
    "starcoder": 8192,
    "starcoder2": 16384,
    "codegemma": 8192,
    "command-r": 131072,
    "yi": 4096,
    "internlm": 8192,
    "solar": 4096,
    "tinyllama": 2048,
    "orca-mini": 4096,
    "neural-chat": 4096,
    "stable-code": 16384,
}


def _detect_hardware() -> HardwareInfo:
    """Detect local hardware capabilities."""
    info = HardwareInfo(
        os_platform=platform.system(),
        cpu_cores=os.cpu_count(),
    )

    # RAM detection
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulong),
                    ("ullAvailPhys", c_ulong),
                    ("ullTotalPageFile", c_ulong),
                    ("ullAvailPageFile", c_ulong),
                    ("ullTotalVirtual", c_ulong),
                    ("ullAvailVirtual", c_ulong),
                    ("ullAvailExtendedVirtual", c_ulong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info.ram_total_mb = int(stat.ullTotalPhys / (1024 * 1024))
        else:
            import shutil
            total, _, _ = shutil.disk_usage("/")
            # On Unix, try reading /proc/meminfo
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            info.ram_total_mb = kb // 1024
                            break
            except FileNotFoundError:
                pass
    except Exception as e:
        logger.debug("Could not detect RAM: %s", e)

    # GPU detection (best-effort via environment variables Ollama sets)
    try:
        # Check common GPU indicators
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None:
            info.gpu_available = True
            info.gpu_name = os.environ.get("NVIDIA_GPU_NAME", "CUDA GPU")

        # Check for ROCm (AMD)
        rocm = os.environ.get("ROCM_PATH") or os.environ.get("HIP_VISIBLE_DEVICES")
        if rocm:
            info.gpu_available = True
            info.gpu_name = info.gpu_name or "AMD ROCm GPU"
    except Exception:
        pass

    return info


def _infer_context_length(family: str, name: str) -> int:
    """Infer context length from model family and name."""
    # Check exact family match
    family_lower = family.lower().strip()
    if family_lower in KNOWN_CONTEXT_LENGTHS:
        return KNOWN_CONTEXT_LENGTHS[family_lower]

    # Check by model name prefix
    name_lower = name.lower().split(":")[0]
    for key, length in KNOWN_CONTEXT_LENGTHS.items():
        if name_lower.startswith(key):
            return length

    # Conservative default
    return 4096


def discover_models(ollama_client: OllamaClient) -> List[ModelProfile]:
    """Discover all locally installed Ollama models and build profiles.

    Queries the Ollama API for installed models, then enriches each
    with inferred context lengths and hardware information.

    Returns:
        List of ModelProfile objects, sorted by size (smallest first).
    """
    try:
        models = ollama_client.list_models()
    except Exception as e:
        logger.error("Model discovery failed: %s", e)
        return []

    profiles: List[ModelProfile] = []

    for m in models:
        family = "unknown"
        param_size = None
        quantization = None

        if m.details:
            family = m.details.family or "unknown"
            param_size = m.details.parameter_size
            quantization = m.details.quantization_level

        context_length = _infer_context_length(family, m.name)

        # Try to get context length from Ollama's show endpoint
        ctx_from_api = _fetch_context_length(ollama_client, m.name)
        if ctx_from_api:
            context_length = ctx_from_api

        profile = ModelProfile(
            name=m.name,
            family=family,
            parameter_size=param_size,
            quantization=quantization,
            size_bytes=m.size_bytes,
            size_human=m.size_human,
            context_length=context_length,
        )
        profiles.append(profile)

    # Sort smallest first (cheapest/fastest preference)
    profiles.sort(key=lambda p: p.size_bytes)

    logger.info("Discovered %d local model(s): %s", len(profiles), [p.name for p in profiles])
    return profiles


def _fetch_context_length(client: OllamaClient, model_name: str) -> Optional[int]:
    """Try to get the actual context length from Ollama's /api/show endpoint."""
    try:
        with client._get_http_client(timeout=5.0) as http:
            resp = http.post("/api/show", json={"name": model_name})
            if resp.status_code == 200:
                data = resp.json()
                # Check modelfile parameters
                model_info = data.get("model_info", {})
                # Look for context length in various locations
                for key in model_info:
                    if "context_length" in key.lower():
                        val = model_info[key]
                        if isinstance(val, (int, float)) and val > 0:
                            return int(val)
                # Check parameters string
                params = data.get("parameters", "")
                if "num_ctx" in params:
                    for line in params.split("\n"):
                        if "num_ctx" in line:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    return int(parts[-1])
                                except ValueError:
                                    pass
    except Exception:
        pass
    return None


def get_hardware_info() -> HardwareInfo:
    """Get cached hardware information for the local system."""
    return _detect_hardware()
