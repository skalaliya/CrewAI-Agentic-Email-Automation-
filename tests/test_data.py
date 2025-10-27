"""Tests for data downloader."""

import pandas as pd
import pytest

from mail_agents.data import DataDownloader


def test_data_downloader_initialization(tmp_path):
    """Test DataDownloader initialization."""
    downloader = DataDownloader(tmp_path)

    assert downloader.data_dir == tmp_path
    assert downloader.dataset_name == "ashfakyeafi/spam-email-classification"


def test_prepare_data_v1_v2_format():
    """Test data preparation with v1/v2 column format."""
    df = pd.DataFrame(
        {
            "v1": ["ham", "spam", "ham", "spam"],
            "v2": ["Hello", "Win money", "Meeting at 3pm", "Click now"],
        }
    )

    downloader = DataDownloader()
    X, y = downloader.prepare_data(df)

    assert len(X) == 4
    assert len(y) == 4
    assert all(isinstance(val, (int, float)) for val in y)


def test_prepare_data_text_label_format():
    """Test data preparation with text/label column format."""
    df = pd.DataFrame(
        {
            "text": ["Hello", "Win money", "Meeting at 3pm", "Click now"],
            "label": ["ham", "spam", "ham", "spam"],
        }
    )

    downloader = DataDownloader()
    X, y = downloader.prepare_data(df)

    assert len(X) == 4
    assert len(y) == 4
    assert list(y) == [0, 1, 0, 1]


def test_prepare_data_with_nulls():
    """Test data preparation handles null values."""
    df = pd.DataFrame(
        {
            "text": ["Hello", None, "Meeting", "Click"],
            "label": ["ham", "spam", None, "spam"],
        }
    )

    downloader = DataDownloader()
    X, y = downloader.prepare_data(df)

    assert len(X) == 4
    assert len(y) == 4
    # Nulls should be filled
    assert all(isinstance(val, str) for val in X)
