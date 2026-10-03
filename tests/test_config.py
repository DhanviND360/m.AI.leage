"""Tests for configuration schemas and persistence."""

from pathlib import Path
from mileage.core.config import MileageConfig, OllamaSettings, WorkspaceSettings


def test_default_config():
    config = MileageConfig()
    assert config.version == "0.1.0"
    assert config.ollama.host == "http://127.0.0.1:11434"
    assert config.workspace.project_name == "mileage-project"
    assert ".git" in config.workspace.ignore_patterns


def test_config_save_and_load(tmp_path: Path):
    config = MileageConfig(
        workspace=WorkspaceSettings(project_name="custom-app"),
        ollama=OllamaSettings(default_model="mistral:latest"),
    )
    saved_file = config.save_to_dir(tmp_path)
    assert saved_file.exists()

    loaded = MileageConfig.load_from_dir(tmp_path)
    assert loaded.workspace.project_name == "custom-app"
    assert loaded.ollama.default_model == "mistral:latest"
