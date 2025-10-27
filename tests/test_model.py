"""Tests for spam classification model."""

import numpy as np
import pandas as pd
import pytest

from mail_agents.model import SpamClassifier


@pytest.fixture
def sample_data():
    """Create sample training data."""
    texts = [
        "Congratulations! You've won a free iPhone. Click here now!",
        "Hey, want to grab lunch tomorrow?",
        "URGENT: Your account will be closed. Act now!",
        "Meeting scheduled for 3pm in conference room A",
        "Get rich quick with this amazing opportunity!!!",
        "Can you review the attached document?",
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham

    return pd.Series(texts), pd.Series(labels)


def test_spam_classifier_initialization(tmp_path):
    """Test SpamClassifier initialization."""
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    classifier = SpamClassifier(model_path, vectorizer_path)

    assert classifier.model_path == model_path
    assert classifier.vectorizer_path == vectorizer_path
    assert classifier.model is None
    assert classifier.vectorizer is None


def test_spam_classifier_train(sample_data, tmp_path):
    """Test model training."""
    X, y = sample_data
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    classifier = SpamClassifier(model_path, vectorizer_path)
    metrics = classifier.train(X, y, test_size=0.3, random_state=42)

    assert "accuracy" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert classifier.model is not None
    assert classifier.vectorizer is not None


def test_spam_classifier_predict(sample_data, tmp_path):
    """Test prediction on new data."""
    X, y = sample_data
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    classifier = SpamClassifier(model_path, vectorizer_path)
    classifier.train(X, y, test_size=0.3, random_state=42)

    # Test predictions
    test_texts = [
        "Free money! Click now!",
        "Let's meet for coffee",
    ]
    predictions = classifier.predict(test_texts)

    assert len(predictions) == 2
    assert all("prediction" in p for p in predictions)
    assert all("is_spam" in p for p in predictions)
    assert all("confidence" in p for p in predictions)


def test_spam_classifier_save_load(sample_data, tmp_path):
    """Test saving and loading model."""
    X, y = sample_data
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    # Train and save
    classifier1 = SpamClassifier(model_path, vectorizer_path)
    classifier1.train(X, y, test_size=0.3, random_state=42)

    # Load in new instance
    classifier2 = SpamClassifier(model_path, vectorizer_path)
    classifier2.load()

    # Test that loaded model works
    test_text = ["Free money!"]
    pred1 = classifier1.predict(test_text)
    pred2 = classifier2.predict(test_text)

    assert pred1[0]["prediction"] == pred2[0]["prediction"]


def test_spam_classifier_predict_without_training(tmp_path):
    """Test that prediction fails without training."""
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    classifier = SpamClassifier(model_path, vectorizer_path)

    with pytest.raises(ValueError, match="Model not trained or loaded"):
        classifier.predict(["test"])


def test_spam_classifier_evaluate(sample_data, tmp_path):
    """Test model evaluation."""
    X, y = sample_data
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    classifier = SpamClassifier(model_path, vectorizer_path)
    classifier.train(X, y, test_size=0.3, random_state=42)

    # Evaluate on same data
    metrics = classifier.evaluate(X, y)

    assert "accuracy" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert 0 <= metrics["accuracy"] <= 1
