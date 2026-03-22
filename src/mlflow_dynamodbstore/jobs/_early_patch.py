"""Early-startup patch for DynamoDB job store support.

This module is loaded at Python interpreter startup via a .pth file,
before any user code runs. It patches:

1. ``mlflow.server.jobs.utils._check_requirements`` — to accept dynamodb://
   URIs, so ``mlflow server`` launches the Huey job runner automatically.
2. ``mlflow.server.handlers._get_job_store`` — to return DynamoDBJobStore
   for dynamodb:// URIs instead of the hardcoded SqlAlchemyJobStore.

Both patches are lazy — they only activate when the backend store URI
starts with dynamodb://, and only import DynamoDBJobStore on first call.
"""

from __future__ import annotations

import os


def _install_patches() -> None:
    store_uri = os.environ.get("_MLFLOW_SERVER_FILE_STORE", "") or os.environ.get(
        "MLFLOW_BACKEND_STORE_URI", ""
    )
    if not store_uri.startswith("dynamodb://"):
        return

    # Patch _check_requirements to accept dynamodb:// URIs
    try:
        import mlflow.server.jobs.utils as jobs_utils

        _original_check = jobs_utils._check_requirements

        def _patched_check_requirements(backend_store_uri: str | None = None) -> None:
            resolved = backend_store_uri or store_uri
            if resolved.startswith("dynamodb://"):
                return  # DynamoDB is a valid backend
            _original_check(backend_store_uri)

        jobs_utils._check_requirements = _patched_check_requirements
    except ImportError:
        pass

    # Patch _get_job_store to return DynamoDBJobStore
    try:
        import mlflow.server.handlers as handlers_module

        _original_get = handlers_module._get_job_store
        _cached_store: dict[str, object] = {}

        def _patched_get_job_store(backend_store_uri: str | None = None) -> object:
            resolved = backend_store_uri or store_uri
            if resolved.startswith("dynamodb://"):
                if "store" not in _cached_store:
                    from mlflow_dynamodbstore.job_store import DynamoDBJobStore

                    _cached_store["store"] = DynamoDBJobStore(resolved)
                return _cached_store["store"]
            return _original_get(backend_store_uri)

        handlers_module._get_job_store = _patched_get_job_store  # type: ignore[assignment]
    except ImportError:
        pass


_install_patches()
