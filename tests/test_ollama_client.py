"""Tests for Ollama client interactions and model verification."""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from mileage.core.exceptions import ModelNotFoundError, OllamaConnectionError
from mileage.models.ollama_client import OllamaClient


def test_ollama_health_offline():
    client = OllamaClient(host="http://127.0.0.1:99999", timeout_seconds=0.5)
    status = client.check_health(quick_timeout=0.2)
    assert not status.is_running
    assert status.error_message is not None


@patch("httpx.Client.get")
def test_ollama_health_online(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"version": "0.3.12"}
    mock_get.return_value = mock_resp

    client = OllamaClient()
    status = client.check_health()
    assert status.is_running
    assert status.version == "0.3.12"


@patch("httpx.Client.get")
def test_list_and_verify_models(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {
                "name": "llama3.2:latest",
                "model": "llama3.2:latest",
                "size": 2048000000,
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "parameter_size": "3B",
                },
            }
        ]
    }
    mock_get.return_value = mock_resp

    client = OllamaClient()
    models = client.list_models()
    assert len(models) == 1
    assert models[0].name == "llama3.2:latest"
    assert models[0].details.parameter_size == "3B"

    assert client.has_model("llama3.2:latest")
    assert client.has_model("llama3.2")
    assert not client.has_model("nonexistent-model")
