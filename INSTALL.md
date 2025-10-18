# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Kaggle account with API credentials
- OpenAI API key

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/skalaliya/CrewAI-Agentic-Email-Automation-.git
cd CrewAI-Agentic-Email-Automation-
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Package

```bash
pip install -e ".[dev]"
```

If you encounter timeout issues with PyPI, try:

```bash
pip install --default-timeout=300 -e ".[dev]"
```

Or install in steps:

```bash
# Install build dependencies
pip install setuptools wheel

# Install core dependencies
pip install scikit-learn pandas numpy joblib

# Install other dependencies
pip install crewai fastapi uvicorn typer python-dotenv kagglehub pyyaml pydantic

# Install dev dependencies
pip install pytest pytest-cov pytest-asyncio httpx black ruff mypy
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
OPENAI_API_KEY=sk-your-openai-api-key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### 5. Verify Installation

```bash
mailai --help
```

You should see the CLI help message.

## Kaggle API Setup

To download the dataset, you need Kaggle API credentials:

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json` with your credentials
5. Add the username and key to your `.env` file

## OpenAI API Setup

CrewAI agents require an OpenAI API key:

1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add it to your `.env` file

## Next Steps

After installation, follow the [README Quick Start](README.md#quick-start) guide to:

1. Download the dataset: `mailai download`
2. Train the model: `mailai train`
3. Run predictions: `mailai predict "email text"`
4. Start the API: `mailai run-api`
