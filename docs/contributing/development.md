# Development

This guide covers setting up a development environment, running tests, and
contributing changes to mlflow-dynamodbstore.

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) for package management
- Docker (for the moto server used by integration tests)
- Git with pre-commit support

## Getting Started

Clone the repository and install all development dependencies:

```bash
git clone https://github.com/zeroae/mlflow-dynamodbstore.git
cd mlflow-dynamodbstore
uv sync --extra dev --extra docs
```

Install the pre-commit hooks so that every commit is automatically checked:

```bash
uv run pre-commit install
```

## Pre-commit Hooks

The project uses three hooks that run on every commit:

| Hook            | Purpose                          |
|-----------------|----------------------------------|
| **ruff**        | Lint and auto-fix Python code    |
| **ruff-format** | Format Python code               |
| **mypy**        | Static type checking on `src/`   |

You can run them manually against all files at any time:

```bash
uv run pre-commit run --all-files
```

## Testing

Tests are organized by scope and marked with pytest markers.

### Test Markers

| Marker            | Scope                                  | Backend            |
|-------------------|----------------------------------------|---------------------|
| `unit`            | Individual functions and classes        | moto `@mock_aws`    |
| `integration`     | End-to-end store operations            | moto server          |
| `compatibility`   | MLflow's own test suite against plugin | moto server          |
| `smoke`           | Real DynamoDB (needs AWS credentials)  | AWS DynamoDB         |

### Running Tests

```bash
# Unit tests only
uv run pytest tests/unit/ -v

# Integration tests only
uv run pytest tests/integration/ -v

# MLflow compatibility tests
uv run pytest tests/compatibility/ -v

# All tests with a specific marker
uv run pytest -m unit -v

# With coverage
uv run pytest tests/unit/ --cov=mlflow_dynamodbstore --cov-report=term-missing
```

### Unit Tests

Unit tests use moto's `@mock_aws` decorator, which patches boto3 in-process.
A shared `mock_dynamodb` fixture is defined in `tests/unit/conftest.py`:

```python
import pytest
from moto import mock_aws

@pytest.fixture
def mock_dynamodb():
    with mock_aws():
        yield
```

Store fixtures (`tracking_store`, `registry_store`, `workspace_store`) build on
top of `mock_dynamodb` and create a fully provisioned table for each test.

### Integration Tests

Integration tests run against a moto server (a standalone HTTP process that
emulates the DynamoDB API). The server is started automatically by the
`tests/integration/conftest.py` fixtures. These tests exercise the full
request-response cycle including HTTP serialization.

### Compatibility Tests

Compatibility tests import and run MLflow's own test suites for the tracking
store and model registry, verifying that the DynamoDB plugin passes the same
assertions as the built-in SQL backend.

## Building Documentation

The project uses MkDocs with the Material theme. To preview the docs locally:

```bash
uv run mkdocs serve
```

To build a static site (with strict warnings):

```bash
uv run mkdocs build --strict
```

## Code Style

- **Line length:** 100 characters
- **Target version:** Python 3.11
- **Linting rules:** `E`, `F`, `I`, `N`, `W`, `UP` (see `[tool.ruff.lint]` in `pyproject.toml`)
- **Type checking:** strict mypy on `src/` (tests are excluded)

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix       | When to use                          |
|--------------|--------------------------------------|
| `feat:`      | New feature                          |
| `fix:`       | Bug fix                              |
| `test:`      | Adding or updating tests             |
| `docs:`      | Documentation changes                |
| `ci:`        | CI/CD configuration                  |
| `refactor:`  | Code restructuring without behavior change |

Examples:

```
feat: add TTL policy support for soft-deleted runs
fix: handle empty trace name in LSI4 sort key
docs: add contributing guide and architecture decision record
```

## Pull Request Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-feature main
   ```
2. **Make your changes** and ensure all pre-commit hooks pass.
3. **Add or update tests** for any new behavior.
4. **Push and open a PR** against `main`:
   ```bash
   git push -u origin feat/my-feature
   gh pr create
   ```
5. **Address review feedback** and keep commits clean.
6. **Merge** once approved and CI is green.

## Project Layout

```
src/mlflow_dynamodbstore/
    dynamodb/           # DynamoDB table abstraction and schema
    auth/               # Authentication plugin
    cli.py              # CLI commands (mlflow-dynamodbstore)
    tracking_store.py   # MLflow tracking store plugin
    registry_store.py   # MLflow model registry plugin
    workspace_store.py  # Workspace provider plugin
tests/
    unit/               # Fast in-process tests (moto decorator)
    integration/        # Server-based tests (moto server)
    compatibility/      # MLflow's own test suite
docs/                   # MkDocs documentation source
```
