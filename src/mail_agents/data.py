"""Kaggle dataset downloader for spam email classification."""

import shutil
from pathlib import Path
from typing import Optional

import kagglehub
import pandas as pd


class DataDownloader:
    """Downloads and prepares the spam email classification dataset from Kaggle."""

    def __init__(self, data_dir: Path = Path("data")):
        """Initialize the data downloader.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.dataset_name = "ashfakyeafi/spam-email-classification"

    def download(self) -> Path:
        """Download the dataset from Kaggle.

        Returns:
            Path to the downloaded dataset directory
        """
        print(f"Downloading dataset: {self.dataset_name}")

        # Download using kagglehub
        download_path = kagglehub.dataset_download(self.dataset_name)
        print(f"Dataset downloaded to: {download_path}")

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Copy files to our data directory
        download_path_obj = Path(download_path)
        for file in download_path_obj.glob("*"):
            if file.is_file():
                dest = self.data_dir / file.name
                shutil.copy2(file, dest)
                print(f"Copied {file.name} to {dest}")

        return self.data_dir

    def load_data(self, file_name: Optional[str] = None) -> pd.DataFrame:
        """Load the dataset into a pandas DataFrame.

        Args:
            file_name: Specific CSV file to load. If None, looks for common names.

        Returns:
            DataFrame containing the email data
        """
        if file_name:
            file_path = self.data_dir / file_name
        else:
            # Try common file names
            possible_names = ["spam.csv", "emails.csv", "spam_email.csv"]
            file_path = None
            for name in possible_names:
                candidate = self.data_dir / name
                if candidate.exists():
                    file_path = candidate
                    break

            if file_path is None:
                # Get the first CSV file
                csv_files = list(self.data_dir.glob("*.csv"))
                if not csv_files:
                    raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
                file_path = csv_files[0]

        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
        return df

    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training.

        Args:
            df: Raw dataframe from Kaggle

        Returns:
            Tuple of (features, labels)
        """
        # The dataset typically has columns like 'text' and 'label' or 'v1' and 'v2'
        # We need to identify the text and label columns
        if "text" in df.columns and "label" in df.columns:
            X = df["text"]
            y = df["label"]
        elif "v2" in df.columns and "v1" in df.columns:
            # Common format for this dataset
            X = df["v2"]
            y = df["v1"]
        elif "email" in df.columns and "spam" in df.columns:
            X = df["email"]
            y = df["spam"]
        else:
            # Use first two columns as fallback
            X = df.iloc[:, 1]
            y = df.iloc[:, 0]

        # Clean the data
        X = X.fillna("")
        y = y.fillna("ham")

        # Convert labels to binary (1 for spam, 0 for ham)
        if y.dtype == object:
            y = y.map(lambda x: 1 if str(x).lower() in ["spam", "1"] else 0)

        return X, y
