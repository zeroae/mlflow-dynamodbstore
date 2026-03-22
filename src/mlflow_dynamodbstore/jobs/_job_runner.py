"""Custom job runner that patches _get_job_store for DynamoDB before
delegating to MLflow's standard job runner.

This module is launched as a subprocess from create_app. It patches
mlflow.server.handlers._get_job_store to return DynamoDBJobStore for
dynamodb:// URIs, then runs MLflow's job runner logic which starts
Huey consumers and enqueues unfinished jobs.
"""

from __future__ import annotations

import logging
import os
import time

_logger = logging.getLogger(__name__)


def _patch_get_job_store() -> None:
    """Patch _get_job_store to handle dynamodb:// URIs."""
    import mlflow.server.handlers as handlers_module

    from mlflow_dynamodbstore.job_store import DynamoDBJobStore

    store_uri = os.environ.get("_MLFLOW_SERVER_FILE_STORE", "") or os.environ.get(
        "MLFLOW_BACKEND_STORE_URI", ""
    )
    if not store_uri.startswith("dynamodb://"):
        return

    _dynamodb_job_store = DynamoDBJobStore(store_uri)
    _original = handlers_module._get_job_store

    def _patched(backend_store_uri: str | None = None) -> DynamoDBJobStore:
        resolved = backend_store_uri or store_uri
        if resolved.startswith("dynamodb://"):
            return _dynamodb_job_store
        return _original(backend_store_uri)  # type: ignore[return-value]

    handlers_module._get_job_store = _patched
    _logger.info("Job runner: patched _get_job_store for DynamoDB.")


if __name__ == "__main__":
    # Patch before importing any MLflow job runner code
    _patch_get_job_store()

    # Now run MLflow's standard job runner logic
    from mlflow.server import HUEY_STORAGE_PATH_ENV_VAR  # type: ignore[attr-defined]
    from mlflow.server.jobs.utils import (
        _enqueue_unfinished_jobs,
        _job_name_to_fn_fullname_map,
        _launch_huey_consumer,
        _launch_periodic_tasks_consumer,
        _start_watcher_to_kill_job_runner_if_mlflow_server_dies,
    )

    logger = logging.getLogger("mlflow_dynamodbstore.jobs._job_runner")
    server_up_time = int(time.time() * 1000)
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies()

    huey_store_path = os.environ[HUEY_STORAGE_PATH_ENV_VAR]

    for job_name in _job_name_to_fn_fullname_map:
        try:
            _launch_huey_consumer(job_name)
        except Exception as e:
            logging.warning(f"Launch Huey consumer for {job_name} jobs failed, root cause: {e!r}")

    _launch_periodic_tasks_consumer()

    time.sleep(10)  # wait for huey consumer launching
    _enqueue_unfinished_jobs(server_up_time)

    # Keep running — the watcher thread monitors the parent server process
    while True:
        time.sleep(60)
