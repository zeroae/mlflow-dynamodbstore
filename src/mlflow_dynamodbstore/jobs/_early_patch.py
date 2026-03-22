"""Early-startup patch for DynamoDB job store support.

This module is loaded at Python interpreter startup via a .pth file,
before any user code runs. It patches mlflow.server.handlers._get_job_store
to return DynamoDBJobStore for dynamodb:// URIs, enabling Huey background
workers in all processes (Flask workers, job runner, Huey consumers).

The patch is lazy — it only activates when the backend store URI starts
with dynamodb://, and only imports DynamoDBJobStore on first call.
"""

from __future__ import annotations

import os


def _install_job_store_patch() -> None:
    store_uri = os.environ.get("_MLFLOW_SERVER_FILE_STORE", "") or os.environ.get(
        "MLFLOW_BACKEND_STORE_URI", ""
    )
    if not store_uri.startswith("dynamodb://"):
        return

    try:
        import mlflow.server.handlers as handlers_module
    except ImportError:
        return

    _original = handlers_module._get_job_store
    _cached_store: dict[str, object] = {}

    def _patched_get_job_store(backend_store_uri: str | None = None) -> object:
        resolved = backend_store_uri or store_uri
        if resolved.startswith("dynamodb://"):
            if "store" not in _cached_store:
                from mlflow_dynamodbstore.job_store import DynamoDBJobStore

                _cached_store["store"] = DynamoDBJobStore(resolved)
            return _cached_store["store"]
        return _original(backend_store_uri)

    handlers_module._get_job_store = _patched_get_job_store  # type: ignore[assignment]


_install_job_store_patch()
