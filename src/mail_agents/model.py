"""Machine learning model for spam classification using TF-IDF and LinearSVC."""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


class SpamClassifier:
    """TF-IDF + LinearSVC baseline model for spam classification."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
    ):
        """Initialize the spam classifier.

        Args:
            model_path: Path to save/load the trained model
            vectorizer_path: Path to save/load the TF-IDF vectorizer
        """
        self.model_path = model_path or Path("models/spam_classifier.pkl")
        self.vectorizer_path = vectorizer_path or Path("models/tfidf_vectorizer.pkl")
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LinearSVC] = None

    def train(
        self,
        X: pd.Series,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """Train the spam classifier.

        Args:
            X: Email text data
            y: Labels (0 for ham, 1 for spam)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing training metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Create and fit TF-IDF vectorizer
        print("Creating TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train LinearSVC model
        print("Training LinearSVC model...")
        self.model = LinearSVC(random_state=random_state, max_iter=2000)
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
        print("\nConfusion Matrix:")
        print(cm)

        # Save the model and vectorizer
        self.save()

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict(self, texts: list[str]) -> list[dict]:
        """Predict spam/ham for given texts.

        Args:
            texts: List of email texts to classify

        Returns:
            List of predictions with probabilities
        """
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        # Transform texts
        X_tfidf = self.vectorizer.transform(texts)

        # Predict
        predictions = self.model.predict(X_tfidf)
        # Get decision function scores (distance from hyperplane)
        scores = self.model.decision_function(X_tfidf)

        # Convert scores to pseudo-probabilities (sigmoid)
        probabilities = 1 / (1 + np.exp(-scores))

        results = []
        for i, text in enumerate(texts):
            results.append(
                {
                    "text": text,
                    "prediction": "spam" if predictions[i] == 1 else "ham",
                    "is_spam": bool(predictions[i] == 1),
                    "confidence": float(probabilities[i] if predictions[i] == 1 else 1 - probabilities[i]),
                }
            )

        return results

    def evaluate(self, X: pd.Series, y: pd.Series) -> dict:
        """Evaluate the model on given data.

        Args:
            X: Email text data
            y: True labels

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        # Transform and predict
        X_tfidf = self.vectorizer.transform(X)
        y_pred = self.model.predict(X_tfidf)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    def save(self):
        """Save the trained model and vectorizer to disk."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("No model to save. Train the model first.")

        # Create models directory
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and vectorizer
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"Model saved to {self.model_path}")
        print(f"Vectorizer saved to {self.vectorizer_path}")

    def load(self):
        """Load a trained model and vectorizer from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {self.vectorizer_path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        print(f"Model loaded from {self.model_path}")
        print(f"Vectorizer loaded from {self.vectorizer_path}")
