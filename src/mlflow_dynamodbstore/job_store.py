"""DynamoDB-backed MLflow job store."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

from botocore.exceptions import ClientError
from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.jobs.abstract_store import AbstractJobStore
from mlflow.utils.time import get_current_time_millis

from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI1_JOB_STATUS_PREFIX,
    GSI1_PK,
    GSI1_SK,
    GSI2_JOBS_PREFIX,
    GSI2_PK,
    GSI2_SK,
    PK_JOB_PREFIX,
    SK_JOB_META,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri

_LIST_JOB_PAGE_SIZE = 100


def _item_to_job(item: dict[str, Any]) -> Job:
    return Job(  # type: ignore[abstract]
        job_id=item["job_id"],
        creation_time=int(item["creation_time"]),
        job_name=item["job_name"],
        params=item.get("params", "{}"),
        timeout=float(item["timeout"]) if item.get("timeout") is not None else None,
        status=JobStatus.from_str(item["status"]),
        result=item.get("result"),
        retry_count=int(item.get("retry_count", 0)),
        last_update_time=int(item["last_update_time"]),
        workspace=item.get("workspace"),
    )


def _job_gsi_attrs(workspace: str, job_id: str, status: str, creation_time: int) -> dict[str, str]:
    """Return all GSI key attributes for a job item."""
    # Sort key: padded creation_time + job_id for stable ordering
    sort_key = f"{creation_time:020d}#{job_id}"
    return {
        GSI2_PK: f"{GSI2_JOBS_PREFIX}{workspace}",
        GSI2_SK: sort_key,
        GSI1_PK: f"{GSI1_JOB_STATUS_PREFIX}{workspace}#{status}",
        GSI1_SK: sort_key,
    }


class DynamoDBJobStore(AbstractJobStore):
    """Job store backed by DynamoDB.

    Each job is a single item:

        PK = JOB#<job_id>
        SK = JOB#META

    Indexed on:
        GSI2: gsi2pk=JOBS#<workspace>, gsi2sk=<job_id> (list all, creation_time ASC)
        GSI1: gsi1pk=JOB_STATUS#<workspace>#<status>, gsi1sk=<job_id> (per-status)
    """

    def __init__(self, store_uri: str) -> None:
        uri = parse_dynamodb_uri(store_uri)
        if uri.deploy:
            ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)

    @property
    def _workspace(self) -> str:
        from mlflow.utils.workspace_context import get_request_workspace
        from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

        return get_request_workspace() or DEFAULT_WORKSPACE_NAME

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_job(self, job_name: str, params: str, timeout: float | None = None) -> Job:
        job_id = str(uuid.uuid4())
        now_ms = get_current_time_millis()
        workspace = self._workspace
        status = JobStatus.PENDING.value

        item: dict[str, Any] = {
            "PK": f"{PK_JOB_PREFIX}{job_id}",
            "SK": SK_JOB_META,
            "job_id": job_id,
            "creation_time": now_ms,
            "job_name": job_name,
            "params": params,
            "status": status,
            "retry_count": 0,
            "last_update_time": now_ms,
            "workspace": workspace,
            **_job_gsi_attrs(workspace, job_id, status, now_ms),
        }
        if timeout is not None:
            item["timeout"] = timeout

        self._table.put_item(item)
        return _item_to_job(item)

    def get_job(self, job_id: str) -> Job:
        item = self._table.get_item(pk=f"{PK_JOB_PREFIX}{job_id}", sk=SK_JOB_META)
        if item is None:
            raise MlflowException(
                f"Job with ID {job_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return _item_to_job(item)

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def _update_job(self, job_id: str, new_status: JobStatus, result: str | None = None) -> Job:
        item = self._table.get_item(pk=f"{PK_JOB_PREFIX}{job_id}", sk=SK_JOB_META)
        if item is None:
            raise MlflowException(
                f"Job with ID {job_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        current = JobStatus.from_str(item["status"])
        if JobStatus.is_finalized(current):
            raise MlflowException(
                f"The Job {job_id} is already finalized with status: {current}, "
                "it can't be updated."
            )

        workspace = item.get("workspace", self._workspace)
        creation_time = int(item["creation_time"])
        now_ms = get_current_time_millis()
        updates: dict[str, Any] = {
            "status": new_status.value,
            "last_update_time": now_ms,
            **_job_gsi_attrs(workspace, job_id, new_status.value, creation_time),
        }
        if result is not None:
            updates["result"] = result

        self._table.update_item(pk=f"{PK_JOB_PREFIX}{job_id}", sk=SK_JOB_META, updates=updates)

        item.update(updates)
        return _item_to_job(item)

    def start_job(self, job_id: str) -> None:
        item = self._table.get_item(pk=f"{PK_JOB_PREFIX}{job_id}", sk=SK_JOB_META)
        if item is None:
            raise MlflowException(
                f"Job with ID {job_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        workspace = item.get("workspace", self._workspace)
        creation_time = int(item["creation_time"])
        now_ms = get_current_time_millis()
        gsi = _job_gsi_attrs(workspace, job_id, JobStatus.RUNNING.value, creation_time)

        try:
            self._table._table.update_item(
                Key={"PK": f"{PK_JOB_PREFIX}{job_id}", "SK": SK_JOB_META},
                UpdateExpression=(
                    "SET #status = :running, #lut = :now, #g1pk = :g1pk, #g1sk = :g1sk"
                ),
                ConditionExpression="#status = :pending",
                ExpressionAttributeNames={
                    "#status": "status",
                    "#lut": "last_update_time",
                    "#g1pk": GSI1_PK,
                    "#g1sk": GSI1_SK,
                },
                ExpressionAttributeValues={
                    ":running": JobStatus.RUNNING.value,
                    ":pending": JobStatus.PENDING.value,
                    ":now": now_ms,
                    ":g1pk": gsi[GSI1_PK],
                    ":g1sk": gsi[GSI1_SK],
                },
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                current = JobStatus.from_str(item["status"])
                raise MlflowException(
                    f"Job {job_id} is in {current} state, cannot start (must be PENDING)"
                )
            raise

    def reset_job(self, job_id: str) -> None:
        self._update_job(job_id, JobStatus.PENDING)

    def finish_job(self, job_id: str, result: str) -> None:
        self._update_job(job_id, JobStatus.SUCCEEDED, result)

    def fail_job(self, job_id: str, error: str) -> None:
        self._update_job(job_id, JobStatus.FAILED, error)

    def mark_job_timed_out(self, job_id: str) -> None:
        self._update_job(job_id, JobStatus.TIMEOUT)

    def retry_or_fail_job(self, job_id: str, error: str) -> int | None:
        from mlflow.environment_variables import MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES

        max_retries = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES.get()

        item = self._table.get_item(pk=f"{PK_JOB_PREFIX}{job_id}", sk=SK_JOB_META)
        if item is None:
            raise MlflowException(
                f"Job with ID {job_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        retry_count = int(item.get("retry_count", 0))
        workspace = item.get("workspace", self._workspace)
        creation_time = int(item["creation_time"])
        now_ms = get_current_time_millis()

        if retry_count >= max_retries:
            updates: dict[str, Any] = {
                "status": JobStatus.FAILED.value,
                "result": error,
                "last_update_time": now_ms,
                **_job_gsi_attrs(workspace, job_id, JobStatus.FAILED.value, creation_time),
            }
            self._table.update_item(pk=f"{PK_JOB_PREFIX}{job_id}", sk=SK_JOB_META, updates=updates)
            return None

        new_count = retry_count + 1
        updates = {
            "retry_count": new_count,
            "status": JobStatus.PENDING.value,
            "last_update_time": now_ms,
            **_job_gsi_attrs(workspace, job_id, JobStatus.PENDING.value, creation_time),
        }
        self._table.update_item(pk=f"{PK_JOB_PREFIX}{job_id}", sk=SK_JOB_META, updates=updates)
        return new_count

    def cancel_job(self, job_id: str) -> Job:
        return self._update_job(job_id, JobStatus.CANCELED)

    # ------------------------------------------------------------------
    # List / Delete
    # ------------------------------------------------------------------

    def list_jobs(
        self,
        job_name: str | None = None,
        statuses: list[JobStatus] | None = None,
        begin_timestamp: int | None = None,
        end_timestamp: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> Iterator[Job]:
        workspace = self._workspace

        if statuses:
            # Query GSI1 per status, merge results
            all_items: list[dict[str, Any]] = []
            for status in statuses:
                gsi1pk = f"{GSI1_JOB_STATUS_PREFIX}{workspace}#{status.value}"
                items = self._table.query(pk=gsi1pk, index_name="gsi1", scan_forward=True)
                all_items.extend(items)
            # Sort by creation_time ASC across status buckets
            all_items.sort(key=lambda i: int(i.get("creation_time", 0)))
        else:
            gsi2pk = f"{GSI2_JOBS_PREFIX}{workspace}"
            all_items = self._table.query(pk=gsi2pk, index_name="gsi2", scan_forward=True)

        for item in all_items:
            if job_name is not None and item.get("job_name") != job_name:
                continue
            ct = int(item.get("creation_time", 0))
            if begin_timestamp is not None and ct < begin_timestamp:
                continue
            if end_timestamp is not None and ct > end_timestamp:
                continue
            if params:
                stored = json.loads(item.get("params", "{}"))
                if not all(stored.get(k) == v for k, v in params.items()):
                    continue
            yield _item_to_job(item)

    def delete_jobs(self, older_than: int = 0, job_ids: list[str] | None = None) -> list[str]:
        current_time = get_current_time_millis()
        time_threshold = current_time - older_than

        # Collect candidate jobs
        workspace = self._workspace
        gsi2pk = f"{GSI2_JOBS_PREFIX}{workspace}"
        all_items = self._table.query(pk=gsi2pk, index_name="gsi2", scan_forward=True)

        ids_to_delete: list[str] = []
        for item in all_items:
            status = JobStatus.from_str(item["status"])
            if not JobStatus.is_finalized(status):
                continue
            jid = item["job_id"]
            if job_ids and jid not in job_ids:
                continue
            if older_than > 0 and int(item["creation_time"]) >= time_threshold:
                continue
            ids_to_delete.append(jid)

        if ids_to_delete:
            keys = [{"PK": f"{PK_JOB_PREFIX}{jid}", "SK": SK_JOB_META} for jid in ids_to_delete]
            self._table.batch_delete(keys)

        return ids_to_delete
