"""Tests for configuration module."""

import os
from pathlib import Path

import pytest

from mail_agents.config import Settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.data_dir == Path("data")
    assert settings.model_path == Path("models/spam_classifier.pkl")
    assert settings.vectorizer_path == Path("models/tfidf_vectorizer.pkl")
    assert settings.config_dir == Path("config")
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.log_level == "INFO"


def test_settings_from_env(monkeypatch):
    """Test settings loaded from environment variables."""
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9000
    assert settings.log_level == "DEBUG"


def test_settings_custom_paths():
    """Test custom path settings."""
    settings = Settings(
        model_path=Path("custom/model.pkl"),
        vectorizer_path=Path("custom/vectorizer.pkl"),
    )

    assert settings.model_path == Path("custom/model.pkl")
    assert settings.vectorizer_path == Path("custom/vectorizer.pkl")
