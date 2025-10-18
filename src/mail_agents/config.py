"""Configuration management for mail agents."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()


class Settings(BaseModel):
    """Application settings."""

    # API Keys
    openai_api_key: Optional[str] = None
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None

    # Paths
    data_dir: Path = Path("data")
    model_path: Path = Path("models/spam_classifier.pkl")
    vectorizer_path: Path = Path("models/tfidf_vectorizer.pkl")
    config_dir: Path = Path("config")

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.kaggle_username = os.getenv("KAGGLE_USERNAME", self.kaggle_username)
        self.kaggle_key = os.getenv("KAGGLE_KEY", self.kaggle_key)
        self.model_path = Path(os.getenv("MODEL_PATH", str(self.model_path)))
        self.vectorizer_path = Path(
            os.getenv("VECTORIZER_PATH", str(self.vectorizer_path))
        )
        self.api_host = os.getenv("API_HOST", self.api_host)
        self.api_port = int(os.getenv("API_PORT", str(self.api_port)))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

    class Config:
        arbitrary_types_allowed = True


settings = Settings()
