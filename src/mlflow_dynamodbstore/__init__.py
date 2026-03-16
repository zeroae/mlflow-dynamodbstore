"""DynamoDB-backed MLflow tracking store, model registry, and auth plugin."""

try:
    from mlflow_dynamodbstore._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"
