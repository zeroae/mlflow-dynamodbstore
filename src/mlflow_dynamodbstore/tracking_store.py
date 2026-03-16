"""DynamoDB-backed MLflow tracking store."""

from __future__ import annotations

import time
from typing import Any

from mlflow.entities import (
    DatasetInput,
    Experiment,
    ExperimentTag,
    Metric,
    Param,
    RunTag,
    ViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking.abstract_store import AbstractStore

from mlflow_dynamodbstore.cache import ResolutionCache
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_EXPERIMENTS_PREFIX,
    GSI2_PK,
    GSI2_SK,
    GSI3_EXP_NAME_PREFIX,
    GSI3_PK,
    GSI3_SK,
    GSI5_EXP_NAMES_PREFIX,
    GSI5_PK,
    GSI5_SK,
    LSI1_SK,
    LSI2_SK,
    LSI3_SK,
    LSI4_SK,
    PK_EXPERIMENT_PREFIX,
    SK_EXPERIMENT_META,
    SK_EXPERIMENT_TAG_PREFIX,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri
from mlflow_dynamodbstore.ids import generate_ulid


def _rev(s: str) -> str:
    """Return reversed string."""
    return s[::-1]


def _item_to_experiment(
    item: dict[str, Any], tags: list[ExperimentTag] | None = None
) -> Experiment:
    """Convert a DynamoDB item to an MLflow Experiment entity."""
    return Experiment(
        experiment_id=item["PK"].replace(PK_EXPERIMENT_PREFIX, ""),
        name=item["name"],
        artifact_location=item.get("artifact_location", ""),
        lifecycle_stage=item.get("lifecycle_stage", "active"),
        tags=tags or [],
        creation_time=item.get("creation_time"),
        last_update_time=item.get("last_update_time"),
    )


class DynamoDBTrackingStore(AbstractStore):
    """MLflow tracking store backed by DynamoDB."""

    def __init__(
        self,
        store_uri: str,
        artifact_uri: str | None = None,
    ) -> None:
        uri = parse_dynamodb_uri(store_uri)
        ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)
        self._cache = ResolutionCache()
        self._artifact_uri = artifact_uri or ""
        self._workspace = "default"
        super().__init__()

    # ------------------------------------------------------------------
    # Experiment CRUD
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        artifact_location: str | None = None,
        tags: list[ExperimentTag] | None = None,
    ) -> str:
        """Create a new experiment and return its ID (ULID)."""
        # Check uniqueness via GSI3
        existing = self._table.query(
            pk=f"{GSI3_EXP_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Experiment(name={name}) already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        now_ms = int(time.time() * 1000)
        exp_id = generate_ulid()

        item: dict[str, Any] = {
            "PK": f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            "SK": SK_EXPERIMENT_META,
            "name": name,
            "lifecycle_stage": "active",
            "artifact_location": artifact_location or self._artifact_uri,
            "creation_time": now_ms,
            "last_update_time": now_ms,
            "workspace": self._workspace,
            # LSI attributes
            LSI1_SK: f"active#{exp_id}",
            LSI2_SK: str(now_ms),
            LSI3_SK: name,
            LSI4_SK: _rev(name),
            # GSI2: list experiments by lifecycle
            GSI2_PK: f"{GSI2_EXPERIMENTS_PREFIX}{self._workspace}#active",
            GSI2_SK: exp_id,
            # GSI3: unique name lookup
            GSI3_PK: f"{GSI3_EXP_NAME_PREFIX}{self._workspace}#{name}",
            GSI3_SK: exp_id,
            # GSI5: all experiment names
            GSI5_PK: f"{GSI5_EXP_NAMES_PREFIX}{self._workspace}",
            GSI5_SK: f"{name}#{exp_id}",
        }

        self._table.put_item(item, condition="attribute_not_exists(PK)")

        # Write tags if provided
        if tags:
            for tag in tags:
                self._write_experiment_tag(exp_id, tag)

        return exp_id

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Fetch an experiment by ID."""
        item = self._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_META,
        )
        if item is None:
            raise MlflowException(
                f"Experiment '{experiment_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        tags = self._get_experiment_tags(experiment_id)
        return _item_to_experiment(item, tags)

    def get_experiment_by_name(self, experiment_name: str) -> Experiment | None:
        """Fetch an experiment by name, or None if not found."""
        results = self._table.query(
            pk=f"{GSI3_EXP_NAME_PREFIX}{self._workspace}#{experiment_name}",
            index_name="gsi3",
            limit=1,
        )
        if not results:
            return None

        exp_id = results[0]["PK"].replace(PK_EXPERIMENT_PREFIX, "")
        return self.get_experiment(exp_id)

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        """Rename an experiment."""
        exp = self.get_experiment(experiment_id)
        old_name = exp.name

        # Check new name uniqueness
        existing = self._table.query(
            pk=f"{GSI3_EXP_NAME_PREFIX}{self._workspace}#{new_name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Experiment(name={new_name}) already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        now_ms = int(time.time() * 1000)

        self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_META,
            updates={
                "name": new_name,
                "last_update_time": now_ms,
                LSI2_SK: str(now_ms),
                LSI3_SK: new_name,
                LSI4_SK: _rev(new_name),
                GSI3_PK: f"{GSI3_EXP_NAME_PREFIX}{self._workspace}#{new_name}",
                GSI5_SK: f"{new_name}#{experiment_id}",
            },
        )

        # Invalidate cache
        self._cache.invalidate("exp_name", old_name)

    def delete_experiment(self, experiment_id: str) -> None:
        """Soft-delete an experiment."""
        now_ms = int(time.time() * 1000)

        self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_META,
            updates={
                "lifecycle_stage": "deleted",
                "last_update_time": now_ms,
                LSI1_SK: f"deleted#{experiment_id}",
                LSI2_SK: str(now_ms),
                GSI2_PK: f"{GSI2_EXPERIMENTS_PREFIX}{self._workspace}#deleted",
            },
        )

    def restore_experiment(self, experiment_id: str) -> None:
        """Restore a soft-deleted experiment."""
        now_ms = int(time.time() * 1000)

        self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_META,
            updates={
                "lifecycle_stage": "active",
                "last_update_time": now_ms,
                LSI1_SK: f"active#{experiment_id}",
                LSI2_SK: str(now_ms),
                GSI2_PK: f"{GSI2_EXPERIMENTS_PREFIX}{self._workspace}#active",
            },
        )

    def search_experiments(
        self,
        view_type: int = ViewType.ACTIVE_ONLY,
        max_results: int = 1000,
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> list[Experiment]:
        """Search experiments using GSI2."""
        if filter_string:
            raise MlflowException(
                "Filter support requires Plan 2",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if order_by:
            raise MlflowException(
                "order_by support requires Plan 2",
                error_code=INVALID_PARAMETER_VALUE,
            )

        experiments: list[Experiment] = []

        if view_type in (ViewType.ACTIVE_ONLY, ViewType.ALL):
            items = self._table.query(
                pk=f"{GSI2_EXPERIMENTS_PREFIX}{self._workspace}#active",
                index_name="gsi2",
                limit=max_results,
            )
            for item in items:
                exp_id = item["PK"].replace(PK_EXPERIMENT_PREFIX, "")
                experiments.append(self.get_experiment(exp_id))

        if view_type in (ViewType.DELETED_ONLY, ViewType.ALL):
            remaining = max_results - len(experiments) if max_results else None
            items = self._table.query(
                pk=f"{GSI2_EXPERIMENTS_PREFIX}{self._workspace}#deleted",
                index_name="gsi2",
                limit=remaining,
            )
            for item in items:
                exp_id = item["PK"].replace(PK_EXPERIMENT_PREFIX, "")
                experiments.append(self.get_experiment(exp_id))

        return experiments[:max_results] if max_results else experiments

    # ------------------------------------------------------------------
    # Experiment tags
    # ------------------------------------------------------------------

    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        """Set a tag on an experiment."""
        # Verify experiment exists
        self.get_experiment(experiment_id)
        self._write_experiment_tag(experiment_id, tag)

    def _write_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        """Write an experiment tag item."""
        item = {
            "PK": f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            "SK": f"{SK_EXPERIMENT_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(item)

    def _get_experiment_tags(self, experiment_id: str) -> list[ExperimentTag]:
        """Read all tags for an experiment."""
        items = self._table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk_prefix=SK_EXPERIMENT_TAG_PREFIX,
        )
        return [ExperimentTag(item["key"], item["value"]) for item in items]

    # ------------------------------------------------------------------
    # Abstract method implementations — stubs for future tasks
    # ------------------------------------------------------------------

    def create_run(
        self,
        experiment_id: str,
        user_id: str,
        start_time: int,
        tags: list[RunTag],
        run_name: str,
    ) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 13")

    def get_run(self, run_id: str) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 13")

    def update_run_info(self, run_id: str, run_status: str, end_time: int, run_name: str) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 13")

    def delete_run(self, run_id: str) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 13")

    def restore_run(self, run_id: str) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 13")

    def _search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str,
        run_view_type: int,
        max_results: int,
        order_by: list[str],
        page_token: str | None,
    ) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 13")

    def log_batch(
        self,
        run_id: str,
        metrics: list[Metric],
        params: list[Param],
        tags: list[RunTag],
    ) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 14")

    def get_metric_history(
        self,
        run_id: str,
        metric_key: str,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 14")

    def log_inputs(
        self,
        run_id: str,
        datasets: list[DatasetInput] | None = None,
        models: Any = None,
    ) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 15")

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        raise MlflowException("Deferred to Plan 3")
