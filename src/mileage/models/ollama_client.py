"""Local Ollama client with auto-launch, robust error handling, verification, and metrics."""

import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
import httpx

from mileage.core.exceptions import (
    ModelNotFoundError,
    OllamaConnectionError,
)
from mileage.core.logger import logger
from mileage.models.schemas import (
    ChatMessage,
    ModelDetails,
    ModelInfo,
    OllamaHealthStatus,
)


class OllamaClient:
    """Interacts directly with a local Ollama instance without cloud reliance."""

    def __init__(self, host: str = "http://127.0.0.1:11434", timeout_seconds: float = 30.0):
        # Normalize host URL
        clean_host = host.rstrip("/")
        if not clean_host.startswith(("http://", "https://")):
            clean_host = f"http://{clean_host}"
        self.host = clean_host
        self.timeout = timeout_seconds

    def _get_http_client(self, timeout: Optional[float] = None) -> httpx.Client:
        """Create an httpx client with specified timeout."""
        return httpx.Client(
            base_url=self.host,
            timeout=timeout or self.timeout,
            follow_redirects=True,
        )

    @classmethod
    def find_ollama_binary(cls) -> Optional[str]:
        """Locate the ollama executable on the local system."""
        # 1. Check system PATH
        path_binary = shutil.which("ollama") or shutil.which("ollama.exe")
        if path_binary and os.path.isfile(path_binary):
            return path_binary

        # 2. Check standard Windows locations
        if sys.platform == "win32":
            home = pathlib.Path.home()
            candidates = [
                home / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
                pathlib.Path("C:/Program Files/Ollama/ollama.exe"),
                pathlib.Path("C:/Program Files (x86)/Ollama/ollama.exe"),
                home / "AppData" / "Local" / "Programs" / "Ollama" / "ollama app.exe",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)

        # 3. Check Unix / macOS standard paths
        for unix_candidate in [
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            "/opt/homebrew/bin/ollama",
        ]:
            if os.path.isfile(unix_candidate):
                return unix_candidate

        return None

    def start_daemon(self, timeout_seconds: float = 12.0) -> bool:
        """
        Automatically launch the local Ollama server if it is not already running.
        Spawns a persistent, detached background process that survives CLI exits.
        """
        health = self.check_health(quick_timeout=1.0)
        if health.is_running:
            return True

        binary = self.find_ollama_binary()
        if not binary:
            logger.warning("Could not automatically locate ollama binary on system.")
            return False

        logger.info("Attempting to auto-start local Ollama daemon using %s", binary)

        try:
            if sys.platform == "win32":
                # Prefer WMI Win32_Process.Create on Windows so daemon survives parent process exits
                clean_bin = binary.replace("'", "''")
                cmd = f"([wmiclass]'win32_process').Create('\"{clean_bin}\" serve')"
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    # Fallback to subprocess DETACHED_PROCESS
                    DETACHED_PROCESS = 0x00000008
                    CREATE_NEW_PROCESS_GROUP = 0x00000200
                    CREATE_NO_WINDOW = 0x08000000
                    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
                    subprocess.Popen(
                        [binary, "serve"],
                        creationflags=flags,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        close_fds=True,
                    )
            else:
                # Unix / macOS daemon
                subprocess.Popen(
                    [binary, "serve"],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
        except Exception as e:
            logger.warning("Failed to spawn Ollama daemon: %s", e)
            return False

        # Wait for daemon to become responsive
        start = time.perf_counter()
        while (time.perf_counter() - start) < timeout_seconds:
            time.sleep(0.3)
            h = self.check_health(quick_timeout=0.8)
            if h.is_running:
                logger.info("Local Ollama daemon is now online at %s", self.host)
                return True

        return False

    def check_health(self, quick_timeout: float = 2.0) -> OllamaHealthStatus:
        """Check if local Ollama service is reachable and responsive."""
        start_time = time.perf_counter()
        try:
            with self._get_http_client(timeout=quick_timeout) as client:
                resp = client.get("/api/version")
                latency_ms = (time.perf_counter() - start_time) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    version = data.get("version", "unknown")
                    # Also count models if possible
                    models_count = 0
                    try:
                        tags_resp = client.get("/api/tags")
                        if tags_resp.status_code == 200:
                            models_count = len(tags_resp.json().get("models", []))
                    except Exception:
                        pass

                    return OllamaHealthStatus(
                        is_running=True,
                        host=self.host,
                        version=version,
                        response_time_ms=round(latency_ms, 2),
                        models_count=models_count,
                    )
                else:
                    return OllamaHealthStatus(
                        is_running=False,
                        host=self.host,
                        error_message=f"HTTP {resp.status_code}: {resp.text}",
                        response_time_ms=round(latency_ms, 2),
                    )
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug("Ollama connection failed: %s", str(e))
            return OllamaHealthStatus(
                is_running=False,
                host=self.host,
                error_message="Could not connect to Ollama. Daemon is likely stopped.",
                response_time_ms=round(latency_ms, 2),
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return OllamaHealthStatus(
                is_running=False,
                host=self.host,
                error_message=f"Unexpected error: {str(e)}",
                response_time_ms=round(latency_ms, 2),
            )

    def list_models(self) -> List[ModelInfo]:
        """Fetch list of all installed local models."""
        try:
            with self._get_http_client() as client:
                resp = client.get("/api/tags")
                if resp.status_code != 200:
                    raise OllamaConnectionError(
                        f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
                raw_models = data.get("models", [])
                result: List[ModelInfo] = []
                for m in raw_models:
                    size = m.get("size", 0)
                    details_raw = m.get("details", {})
                    details = ModelDetails(
                        format=details_raw.get("format"),
                        family=details_raw.get("family"),
                        families=details_raw.get("families"),
                        parameter_size=details_raw.get("parameter_size"),
                        quantization_level=details_raw.get("quantization_level"),
                    )
                    result.append(
                        ModelInfo(
                            name=m.get("name", "unknown"),
                            model=m.get("model", m.get("name", "unknown")),
                            size_bytes=size,
                            size_human=ModelInfo.format_size(size),
                            digest=m.get("digest"),
                            modified_at=m.get("modified_at"),
                            details=details,
                        )
                    )
                return result
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            logger.error("Failed connecting to Ollama tags: %s", e)
            raise OllamaConnectionError() from e
        except Exception as e:
            if isinstance(e, OllamaConnectionError):
                raise
            logger.error("Error retrieving Ollama models: %s", e)
            raise OllamaConnectionError(f"Error querying Ollama API: {str(e)}") from e

    def verify_installed_models(self) -> Tuple[bool, List[ModelInfo]]:
        """Automatically verify if Ollama is running and has installed models."""
        health = self.check_health()
        if not health.is_running:
            return False, []
        try:
            models = self.list_models()
            return len(models) > 0, models
        except Exception:
            return False, []

    def has_model(self, model_name: str) -> bool:
        """Check if a specific model (or model tag) is installed locally."""
        models = self.list_models()
        name_clean = model_name.strip().lower()
        for m in models:
            m_name = m.name.lower()
            if m_name == name_clean:
                return True
            # Also match without tag (e.g. "llama3" matching "llama3:latest")
            if ":" not in name_clean and m_name.split(":")[0] == name_clean:
                return True
        return False

    def resolve_active_model(self, preferred_model: Optional[str] = None) -> str:
        """
        Resolve the active model to use.
        If the preferred model is not installed locally, automatically picks the
        first installed model available (e.g. gemma4:e2b) so execution works right away.
        """
        # Ensure daemon is up first
        self.start_daemon(timeout_seconds=5.0)

        try:
            models = self.list_models()
            if not models:
                return preferred_model or "llama3.2:latest"

            model_names = [m.name for m in models]

            # 1. If preferred model is provided and present, use it
            if preferred_model:
                clean_pref = preferred_model.strip().lower()
                for m in models:
                    if m.name.lower() == clean_pref or m.name.split(":")[0].lower() == clean_pref:
                        return m.name

            # 2. Default requested was not found or not given; select the first installed model!
            fallback = models[0].name
            logger.info("Resolved active local model: %s (installed models: %s)", fallback, model_names)
            return fallback

        except Exception as e:
            logger.debug("Could not resolve model from Ollama API: %s", e)
            return preferred_model or "llama3.2:latest"

    def warmup_model(self, model: str) -> None:
        """Pre-load model weights into local GPU/memory for immediate readiness."""
        try:
            with self._get_http_client(timeout=15.0) as client:
                client.post(
                    "/api/generate",
                    json={
                        "model": model,
                        "prompt": "ready",
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
        except Exception as e:
            logger.debug("Model warmup check: %s", e)

    def ensure_ready(self, preferred_model: Optional[str] = None, auto_warm: bool = False) -> Tuple[bool, str]:
        """
        Guarantees that Ollama is started and an existing local model is selected.
        Returns:
            (is_online, active_model_name)
        """
        is_online = self.start_daemon(timeout_seconds=8.0)
        active_model = self.resolve_active_model(preferred_model)
        if is_online and auto_warm:
            self.warmup_model(active_model)
        return is_online, active_model

    def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Perform non-streaming chat completion."""
        # Ensure daemon is running
        self.start_daemon(timeout_seconds=5.0)

        # Verify model exists first
        if not self.has_model(model):
            # Try to auto-resolve to an existing installed model
            resolved = self.resolve_active_model(model)
            if resolved and resolved != model:
                model = resolved
            else:
                raise ModelNotFoundError(model)

        payload = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }

        start_time = time.perf_counter()
        try:
            with self._get_http_client() as client:
                resp = client.post("/api/chat", json=payload)
                duration_ms = (time.perf_counter() - start_time) * 1000

                if resp.status_code != 200:
                    raise OllamaConnectionError(
                        f"Ollama chat error (HTTP {resp.status_code}): {resp.text}"
                    )

                data = resp.json()
                data["_client_latency_ms"] = round(duration_ms, 2)
                return data
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise OllamaConnectionError() from e

    def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
    ) -> Iterator[Dict[str, Any]]:
        """Stream chat tokens incrementally from local Ollama."""
        # Ensure daemon is running
        self.start_daemon(timeout_seconds=5.0)

        if not self.has_model(model):
            resolved = self.resolve_active_model(model)
            if resolved and resolved != model:
                model = resolved
            else:
                raise ModelNotFoundError(model)

        payload = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            with self._get_http_client() as client:
                with client.stream("POST", "/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        raise OllamaConnectionError(
                            f"Ollama stream error (HTTP {response.status_code})"
                        )
                    for line in response.iter_lines():
                        if not line or not line.strip():
                            continue
                        try:
                            import json
                            chunk = json.loads(line)
                            yield chunk
                        except Exception:
                            continue
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise OllamaConnectionError() from e
