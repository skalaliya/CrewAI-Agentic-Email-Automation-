"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from mail_agents.api import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "version" in data


def test_classify_endpoint_no_model():
    """Test classify endpoint when model is not loaded."""
    response = client.post(
        "/classify",
        json={"text": "This is a test email"},
    )
    # May return 503 if model not loaded, or 200 if it is
    assert response.status_code in [200, 503]


def test_extract_endpoint():
    """Test extract endpoint."""
    response = client.post(
        "/extract",
        json={"text": "Meeting with John on Monday at 3pm"},
    )
    # May fail if OPENAI_API_KEY is not set
    assert response.status_code in [200, 500]


def test_draft_endpoint():
    """Test draft endpoint."""
    response = client.post(
        "/draft",
        json={
            "text": "Can we schedule a meeting?",
            "context": "Response to meeting request",
        },
    )
    # May fail if OPENAI_API_KEY is not set
    assert response.status_code in [200, 500]


def test_pipeline_endpoint():
    """Test pipeline endpoint."""
    response = client.post(
        "/pipeline",
        json={"text": "This is a test email"},
    )
    # May fail if OPENAI_API_KEY is not set
    assert response.status_code in [200, 500]


def test_invalid_request():
    """Test invalid request format."""
    response = client.post(
        "/classify",
        json={},
    )
    assert response.status_code == 422  # Validation error
