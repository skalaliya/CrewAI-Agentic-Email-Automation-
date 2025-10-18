# Mail Agents - Agentic Email Automation with CrewAI

[![CI](https://github.com/skalaliya/CrewAI-Agentic-Email-Automation-/actions/workflows/ci.yml/badge.svg)](https://github.com/skalaliya/CrewAI-Agentic-Email-Automation-/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Python application for agentic email automation using CrewAI. This project combines machine learning (TF-IDF + LinearSVC) for spam classification with intelligent AI agents for email processing, information extraction, and response drafting.

## Features

- 🤖 **AI Agents**: CrewAI-powered agents for email classification, extraction, and drafting
- 📊 **ML Baseline**: TF-IDF + LinearSVC spam classifier with high accuracy
- 🚀 **FastAPI Server**: Production-ready REST API with health checks and multiple endpoints
- 💻 **CLI Tool**: Typer-based command-line interface for all operations
- 📥 **Kaggle Integration**: Automatic dataset download using kagglehub
- 🧪 **Testing**: Comprehensive pytest test suite with coverage
- 🔄 **CI/CD**: GitHub Actions workflow for automated testing
- 📝 **Configuration**: YAML-based agent and task configuration

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Kaggle account (for dataset download)
- OpenAI API key (for CrewAI agents)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/skalaliya/CrewAI-Agentic-Email-Automation-.git
cd CrewAI-Agentic-Email-Automation-
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e ".[dev]"
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY
# - KAGGLE_USERNAME
# - KAGGLE_KEY
```

### Usage

#### 1. Download Dataset

Download the spam email classification dataset from Kaggle:

```bash
mailai download
```

The dataset will be downloaded to the `data/` directory.

#### 2. Train Model

Train the spam classification model:

```bash
mailai train
```

This will:
- Load the dataset
- Train a TF-IDF + LinearSVC model
- Save the model to `models/spam_classifier.pkl`
- Display training metrics and accuracy

#### 3. Evaluate Model

Evaluate the trained model:

```bash
mailai eval
```

#### 4. Predict

Classify a single email:

```bash
mailai predict "Congratulations! You've won a free iPhone. Click here now!"
```

#### 5. Run API Server

Start the FastAPI server:

```bash
mailai run-api
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Classify Email
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Free money! Click now!"}'
```

### Extract Information
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Meeting with John on Monday at 3pm in Room A"}'
```

### Draft Response
```bash
curl -X POST http://localhost:8000/draft \
  -H "Content-Type: application/json" \
  -d '{"text": "Can we schedule a meeting?", "context": "Initial meeting request"}'
```

### Process Pipeline
```bash
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{"text": "Your email text here"}'
```

## CLI Commands

### Download Dataset
```bash
mailai download [--data-dir PATH]
```

### Train Model
```bash
mailai train [--data-dir PATH] [--model-path PATH] [--test-size FLOAT]
```

### Evaluate Model
```bash
mailai eval [--data-dir PATH] [--model-path PATH]
```

### Predict
```bash
mailai predict "Email text" [--model-path PATH]
```

### Run API
```bash
mailai run-api [--host HOST] [--port PORT] [--reload]
```

## Project Structure

```
CrewAI-Agentic-Email-Automation-/
├── src/
│   └── mail_agents/
│       ├── __init__.py          # Package initialization
│       ├── config.py            # Configuration management
│       ├── data.py              # Kaggle dataset downloader
│       ├── model.py             # TF-IDF + LinearSVC classifier
│       ├── agents.py            # CrewAI agents
│       ├── api.py               # FastAPI server
│       └── cli.py               # Typer CLI
├── config/
│   ├── agents.yaml              # Agent configurations
│   └── tasks.yaml               # Task definitions
├── tests/
│   ├── test_config.py           # Config tests
│   ├── test_data.py             # Data tests
│   ├── test_model.py            # Model tests
│   └── test_api.py              # API tests
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
```

### Linting

```bash
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/mail_agents/
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options:

- `OPENAI_API_KEY`: OpenAI API key for CrewAI agents
- `KAGGLE_USERNAME`: Kaggle username for dataset download
- `KAGGLE_KEY`: Kaggle API key for dataset download
- `MODEL_PATH`: Path to save/load the trained model
- `VECTORIZER_PATH`: Path to save/load the TF-IDF vectorizer
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)

### Agent Configuration

Agents are configured in `config/agents.yaml`:

- `spam_classifier`: Spam email classification agent
- `email_extractor`: Information extraction agent
- `email_drafter`: Response drafting agent
- `pipeline_coordinator`: Pipeline orchestration agent

### Task Configuration

Tasks are defined in `config/tasks.yaml`:

- `classify_email`: Email classification task
- `extract_information`: Information extraction task
- `draft_response`: Response drafting task
- `process_pipeline`: Complete pipeline processing task

## Dataset

This project uses the [Spam Email Classification](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification) dataset from Kaggle. The dataset is not included in the repository and must be downloaded using the CLI tool.

**Note**: The dataset is automatically downloaded using kagglehub and is excluded from version control via `.gitignore`.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the agent framework
- [Kaggle](https://www.kaggle.com/) for the spam email dataset
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Typer](https://typer.tiangolo.com/) for the CLI framework