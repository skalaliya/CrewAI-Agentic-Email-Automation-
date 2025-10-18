# Contributing to Mail Agents

Thank you for your interest in contributing to Mail Agents! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/CrewAI-Agentic-Email-Automation-.git
   cd CrewAI-Agentic-Email-Automation-
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

### Formatting

Format your code with Black:

```bash
black src/ tests/
```

### Linting

Check your code with Ruff:

```bash
ruff check src/ tests/
```

Fix auto-fixable issues:

```bash
ruff check --fix src/ tests/
```

### Type Checking

Run MyPy for type checking:

```bash
mypy src/mail_agents/
```

## Testing

We use pytest for testing. All new features should include tests.

### Running Tests

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src/mail_agents --cov-report=term-missing
```

Run specific test file:

```bash
pytest tests/test_model.py
```

Run specific test:

```bash
pytest tests/test_model.py::test_spam_classifier_train
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures for reusable test data
- Aim for high test coverage (>80%)

Example test:

```python
import pytest
from mail_agents.model import SpamClassifier

@pytest.fixture
def sample_data():
    texts = ["spam email", "legitimate email"]
    labels = [1, 0]
    return texts, labels

def test_classifier_prediction(sample_data):
    texts, labels = sample_data
    classifier = SpamClassifier()
    # ... test code
```

## Pull Request Process

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new features
   - Update documentation as needed

3. **Run tests and checks**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   pytest
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template
   - Request review

## Commit Message Guidelines

We follow conventional commit messages:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:

```
feat: add email sentiment analysis
fix: correct spam classification threshold
docs: update API examples in README
test: add tests for email extraction
```

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

Reviewers will check:

- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Security considerations

## Reporting Bugs

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/stack traces
- Relevant code snippets

## Feature Requests

We welcome feature requests! Please include:

- Clear description of the feature
- Use case / motivation
- Proposed implementation (if any)
- Any alternatives considered

## Questions?

If you have questions, please:

- Check the documentation
- Search existing issues
- Open a new issue with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
