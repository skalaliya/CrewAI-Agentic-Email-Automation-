"""Typer CLI for mail agents."""

from pathlib import Path
from typing import Optional

import typer

from .config import settings
from .data import DataDownloader
from .model import SpamClassifier

app = typer.Typer(
    name="mailai",
    help="Mail Agents CLI - Agentic email automation with CrewAI",
    add_completion=False,
)


@app.command()
def download(
    data_dir: str = typer.Option(
        "data",
        "--data-dir",
        "-d",
        help="Directory to store downloaded data",
    ),
):
    """Download the spam email classification dataset from Kaggle."""
    typer.echo("📥 Downloading spam email dataset from Kaggle...")

    try:
        downloader = DataDownloader(Path(data_dir))
        result_path = downloader.download()
        typer.echo(f"✅ Dataset downloaded successfully to: {result_path}")
    except Exception as e:
        typer.echo(f"❌ Error downloading dataset: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def train(
    data_dir: str = typer.Option(
        "data",
        "--data-dir",
        "-d",
        help="Directory containing the dataset",
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to save the trained model",
    ),
    vectorizer_path: Optional[str] = typer.Option(
        None,
        "--vectorizer-path",
        "-v",
        help="Path to save the vectorizer",
    ),
    test_size: float = typer.Option(
        0.2,
        "--test-size",
        "-t",
        help="Proportion of data to use for testing",
    ),
):
    """Train the spam classification model."""
    typer.echo("🚀 Training spam classification model...")

    try:
        # Load data
        downloader = DataDownloader(Path(data_dir))
        df = downloader.load_data()
        X, y = downloader.prepare_data(df)

        typer.echo(f"📊 Dataset loaded: {len(X)} samples")
        typer.echo(f"   Spam: {sum(y)} | Ham: {len(y) - sum(y)}")

        # Train model
        model_path_obj = Path(model_path) if model_path else settings.model_path
        vectorizer_path_obj = Path(vectorizer_path) if vectorizer_path else settings.vectorizer_path

        classifier = SpamClassifier(model_path_obj, vectorizer_path_obj)
        metrics = classifier.train(X, y, test_size=test_size)

        typer.echo(f"\n✅ Model trained successfully!")
        typer.echo(f"   Accuracy: {metrics['accuracy']:.4f}")
        typer.echo(f"   Train samples: {metrics['train_size']}")
        typer.echo(f"   Test samples: {metrics['test_size']}")

    except Exception as e:
        typer.echo(f"❌ Error training model: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def eval(
    data_dir: str = typer.Option(
        "data",
        "--data-dir",
        "-d",
        help="Directory containing the dataset",
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to the trained model",
    ),
    vectorizer_path: Optional[str] = typer.Option(
        None,
        "--vectorizer-path",
        "-v",
        help="Path to the vectorizer",
    ),
):
    """Evaluate the trained model."""
    typer.echo("📊 Evaluating spam classification model...")

    try:
        # Load data
        downloader = DataDownloader(Path(data_dir))
        df = downloader.load_data()
        X, y = downloader.prepare_data(df)

        # Load model
        model_path_obj = Path(model_path) if model_path else settings.model_path
        vectorizer_path_obj = Path(vectorizer_path) if vectorizer_path else settings.vectorizer_path

        classifier = SpamClassifier(model_path_obj, vectorizer_path_obj)
        classifier.load()

        # Evaluate
        metrics = classifier.evaluate(X, y)

        typer.echo(f"\n✅ Evaluation complete!")
        typer.echo(f"   Accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        typer.echo(f"❌ Error evaluating model: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def predict(
    text: str = typer.Argument(..., help="Email text to classify"),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to the trained model",
    ),
    vectorizer_path: Optional[str] = typer.Option(
        None,
        "--vectorizer-path",
        "-v",
        help="Path to the vectorizer",
    ),
):
    """Predict whether an email is spam or ham."""
    typer.echo("🔍 Classifying email...")

    try:
        # Load model
        model_path_obj = Path(model_path) if model_path else settings.model_path
        vectorizer_path_obj = Path(vectorizer_path) if vectorizer_path else settings.vectorizer_path

        classifier = SpamClassifier(model_path_obj, vectorizer_path_obj)
        classifier.load()

        # Predict
        predictions = classifier.predict([text])
        result = predictions[0]

        typer.echo(f"\n✅ Classification Result:")
        typer.echo(f"   Prediction: {result['prediction'].upper()}")
        typer.echo(f"   Confidence: {result['confidence']:.2%}")
        typer.echo(f"   Is Spam: {result['is_spam']}")

    except Exception as e:
        typer.echo(f"❌ Error predicting: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def run_api(
    host: str = typer.Option(
        None,
        "--host",
        "-h",
        help="Host to bind the API server",
    ),
    port: int = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to bind the API server",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
    ),
):
    """Run the FastAPI server."""
    import uvicorn

    from .api import app as fastapi_app

    host = host or settings.api_host
    port = port or settings.api_port

    typer.echo(f"🚀 Starting Mail Agents API server...")
    typer.echo(f"   Host: {host}")
    typer.echo(f"   Port: {port}")
    typer.echo(f"   Docs: http://{host}:{port}/docs")

    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
