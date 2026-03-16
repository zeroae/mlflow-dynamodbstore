"""DynamoDB-backed MLflow tracking store."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from mlflow.entities import (
    DatasetInput,
    Experiment,
    ExperimentTag,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
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
from mlflow_dynamodbstore.dynamodb.keys import pad_step
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI1_PK,
    GSI1_RUN_PREFIX,
    GSI1_SK,
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
    LSI5_SK,
    PK_EXPERIMENT_PREFIX,
    SK_EXPERIMENT_META,
    SK_EXPERIMENT_TAG_PREFIX,
    SK_METRIC_HISTORY_PREFIX,
    SK_METRIC_PREFIX,
    SK_PARAM_PREFIX,
    SK_RANK_PREFIX,
    SK_RUN_PREFIX,
    SK_TAG_PREFIX,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri
from mlflow_dynamodbstore.ids import generate_ulid, ulid_from_timestamp


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


def _item_to_run_info(item: dict[str, Any]) -> RunInfo:
    """Convert a DynamoDB item to an MLflow RunInfo entity."""
    return RunInfo(
        run_id=item["run_id"],
        experiment_id=item["experiment_id"],
        user_id=item.get("user_id", ""),
        status=item.get("status", "RUNNING"),
        start_time=item.get("start_time"),
        end_time=item.get("end_time"),
        lifecycle_stage=item.get("lifecycle_stage", "active"),
        artifact_uri=item.get("artifact_uri", ""),
        run_name=item.get("run_name", ""),
    )


def _item_to_run(
    item: dict[str, Any],
    tags: list[dict[str, Any]],
    params: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> Run:
    """Convert DynamoDB items to an MLflow Run entity."""
    info = _item_to_run_info(item)
    data = RunData(
        tags=[RunTag(t["key"], t["value"]) for t in tags],
        params=[Param(p["key"], p["value"]) for p in params],
        metrics=[
            Metric(m["key"], float(m["value"]), m.get("timestamp", 0), m.get("step", 0))
            for m in metrics
        ],
    )
    return Run(run_info=info, run_data=data)


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

    # ------------------------------------------------------------------
    # Run CRUD
    # ------------------------------------------------------------------

    def create_run(
        self,
        experiment_id: str,
        user_id: str,
        start_time: int,
        tags: list[RunTag],
        run_name: str,
    ) -> Run:
        """Create a new run within an experiment."""
        # Verify experiment exists
        exp = self.get_experiment(experiment_id)

        run_id = ulid_from_timestamp(start_time)
        artifact_uri = f"{exp.artifact_location}/{run_id}/artifacts"

        item: dict[str, Any] = {
            "PK": f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            "SK": f"{SK_RUN_PREFIX}{run_id}",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "status": "RUNNING",
            "start_time": start_time,
            "run_name": run_name,
            "lifecycle_stage": "active",
            "artifact_uri": artifact_uri,
            # LSI attributes
            LSI1_SK: f"active#{run_id}",
            LSI3_SK: f"RUNNING#{run_id}",
            LSI4_SK: run_name.lower() if run_name else "",
            # GSI1: reverse lookup run_id -> experiment_id
            GSI1_PK: f"{GSI1_RUN_PREFIX}{run_id}",
            GSI1_SK: f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
        }

        self._table.put_item(item, condition="attribute_not_exists(PK)")

        # Write tags if provided
        if tags:
            for tag in tags:
                self._write_run_tag(experiment_id, run_id, tag)

        # Cache run_id -> experiment_id
        self._cache.put("run_exp", run_id, experiment_id)

        return self._build_run(item, tags)

    def get_run(self, run_id: str) -> Run:
        """Fetch a run by ID."""
        experiment_id = self._resolve_run_experiment(run_id)

        item = self._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=f"{SK_RUN_PREFIX}{run_id}",
        )
        if item is None:
            raise MlflowException(
                f"Run '{run_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Query tags, params, metrics for this run
        run_prefix = f"{SK_RUN_PREFIX}{run_id}"
        tag_items = self._table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk_prefix=f"{run_prefix}{SK_TAG_PREFIX}",
        )
        param_items = self._table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk_prefix=f"{run_prefix}{SK_PARAM_PREFIX}",
        )
        metric_items = self._table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk_prefix=f"{run_prefix}{SK_METRIC_PREFIX}",
        )

        return _item_to_run(item, tag_items, param_items, metric_items)

    def update_run_info(
        self, run_id: str, run_status: str, end_time: int, run_name: str
    ) -> RunInfo:
        """Update run status, end_time, and run_name."""
        experiment_id = self._resolve_run_experiment(run_id)

        # Get current run to compute duration
        current = self._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=f"{SK_RUN_PREFIX}{run_id}",
        )
        if current is None:
            raise MlflowException(
                f"Run '{run_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        updates: dict[str, Any] = {
            "status": run_status,
            "run_name": run_name,
            LSI3_SK: f"{run_status}#{run_id}",
            LSI4_SK: run_name.lower() if run_name else "",
        }

        if end_time is not None:
            updates["end_time"] = end_time
            updates[LSI2_SK] = str(end_time)
            start_time = current.get("start_time", 0)
            if start_time:
                updates[LSI5_SK] = str(end_time - start_time)

        updated_item = self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=f"{SK_RUN_PREFIX}{run_id}",
            updates=updates,
        )

        assert updated_item is not None  # update always returns ALL_NEW
        return _item_to_run_info(updated_item)

    def delete_run(self, run_id: str) -> None:
        """Soft-delete a run."""
        experiment_id = self._resolve_run_experiment(run_id)

        self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=f"{SK_RUN_PREFIX}{run_id}",
            updates={
                "lifecycle_stage": "deleted",
                LSI1_SK: f"deleted#{run_id}",
            },
        )

    def restore_run(self, run_id: str) -> None:
        """Restore a soft-deleted run."""
        experiment_id = self._resolve_run_experiment(run_id)

        self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=f"{SK_RUN_PREFIX}{run_id}",
            updates={
                "lifecycle_stage": "active",
                LSI1_SK: f"active#{run_id}",
            },
        )

    def _search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str,
        run_view_type: int,
        max_results: int,
        order_by: list[str],
        page_token: str | None,
    ) -> tuple[list[Run], str | None]:
        """Search runs across experiments."""
        if filter_string:
            raise MlflowException(
                "Filter support requires Plan 2",
                error_code=INVALID_PARAMETER_VALUE,
            )

        runs: list[Run] = []

        for exp_id in experiment_ids:
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"

            if run_view_type == ViewType.ACTIVE_ONLY:
                items = self._table.query(
                    pk=pk,
                    sk_prefix="active#",
                    index_name="lsi1",
                    limit=max_results - len(runs) if max_results else None,
                )
            elif run_view_type == ViewType.DELETED_ONLY:
                items = self._table.query(
                    pk=pk,
                    sk_prefix="deleted#",
                    index_name="lsi1",
                    limit=max_results - len(runs) if max_results else None,
                )
            else:
                # ALL: query by SK prefix on main table
                items = self._table.query(
                    pk=pk,
                    sk_prefix=SK_RUN_PREFIX,
                    limit=max_results - len(runs) if max_results else None,
                )

            for item in items:
                # Filter out non-run items (experiment meta, tags, etc.)
                if "run_id" not in item:
                    continue
                run_id = item["run_id"]
                # Cache for later lookups
                self._cache.put("run_exp", run_id, exp_id)
                # Build run with tags/params/metrics
                run_prefix = f"{SK_RUN_PREFIX}{run_id}"
                tag_items = self._table.query(pk=pk, sk_prefix=f"{run_prefix}{SK_TAG_PREFIX}")
                param_items = self._table.query(pk=pk, sk_prefix=f"{run_prefix}{SK_PARAM_PREFIX}")
                metric_items = self._table.query(pk=pk, sk_prefix=f"{run_prefix}{SK_METRIC_PREFIX}")
                runs.append(_item_to_run(item, tag_items, param_items, metric_items))

                if max_results and len(runs) >= max_results:
                    break

            if max_results and len(runs) >= max_results:
                break

        return runs[:max_results] if max_results else runs, None

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------

    def _resolve_run_experiment(self, run_id: str) -> str:
        """Resolve run_id to experiment_id, using cache then GSI1."""
        cached = self._cache.get("run_exp", run_id)
        if cached:
            return cached

        # Look up via GSI1
        results = self._table.query(
            pk=f"{GSI1_RUN_PREFIX}{run_id}",
            index_name="gsi1",
            limit=1,
        )
        if not results:
            raise MlflowException(
                f"Run '{run_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        experiment_id: str = results[0]["experiment_id"]
        self._cache.put("run_exp", run_id, experiment_id)
        return experiment_id

    def _write_run_tag(self, experiment_id: str, run_id: str, tag: RunTag) -> None:
        """Write a run tag item."""
        item = {
            "PK": f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            "SK": f"{SK_RUN_PREFIX}{run_id}{SK_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(item)

    def _build_run(self, item: dict[str, Any], tags: list[RunTag] | None = None) -> Run:
        """Build a Run entity from an item and optional in-memory tags."""
        info = _item_to_run_info(item)
        data = RunData(
            tags=tags or [],
            params=[],
            metrics=[],
        )
        return Run(run_info=info, run_data=data)

    def log_batch(
        self,
        run_id: str,
        metrics: list[Metric],
        params: list[Param],
        tags: list[RunTag],
    ) -> None:
        """Log a batch of metrics, params, and tags for a run."""
        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        items: list[dict[str, Any]] = []

        for metric in metrics:
            # Latest metric item (overwrites previous for same key)
            latest_sk = f"{SK_RUN_PREFIX}{run_id}{SK_METRIC_PREFIX}{metric.key}"
            ddb_value = Decimal(str(metric.value))
            items.append(
                {
                    "PK": pk,
                    "SK": latest_sk,
                    "key": metric.key,
                    "value": ddb_value,
                    "timestamp": metric.timestamp,
                    "step": metric.step,
                }
            )

            # History item (unique per key+step+timestamp)
            padded = pad_step(metric.step)
            hist_sk = (
                f"{SK_RUN_PREFIX}{run_id}{SK_METRIC_HISTORY_PREFIX}"
                f"{metric.key}#{padded}#{metric.timestamp}"
            )
            items.append(
                {
                    "PK": pk,
                    "SK": hist_sk,
                    "key": metric.key,
                    "value": ddb_value,
                    "timestamp": metric.timestamp,
                    "step": metric.step,
                }
            )

            # RANK item for metric (inverted value for descending sort)
            max_val = 9999999999.9999
            inv = max_val - float(metric.value)
            inv_str = f"{inv:020.4f}"
            rank_sk = f"{SK_RANK_PREFIX}m#{metric.key}#{inv_str}#{run_id}"
            items.append(
                {
                    "PK": pk,
                    "SK": rank_sk,
                    "key": metric.key,
                    "value": ddb_value,
                    "run_id": run_id,
                }
            )

        for param in params:
            param_sk = f"{SK_RUN_PREFIX}{run_id}{SK_PARAM_PREFIX}{param.key}"
            items.append(
                {
                    "PK": pk,
                    "SK": param_sk,
                    "key": param.key,
                    "value": param.value,
                }
            )

            # RANK item for param
            rank_sk = f"{SK_RANK_PREFIX}p#{param.key}#{param.value}#{run_id}"
            items.append(
                {
                    "PK": pk,
                    "SK": rank_sk,
                    "key": param.key,
                    "value": param.value,
                    "run_id": run_id,
                }
            )

        # Write all items in batch
        if items:
            self._table.batch_write(items)

        # Write tags individually (uses put_item)
        for tag in tags:
            self._write_run_tag(experiment_id, run_id, tag)

    def get_metric_history(
        self,
        run_id: str,
        metric_key: str,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> list[Metric]:
        """Return the history of a metric for a run, ordered by step."""
        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        prefix = f"{SK_RUN_PREFIX}{run_id}{SK_METRIC_HISTORY_PREFIX}{metric_key}#"

        items = self._table.query(
            pk=pk,
            sk_prefix=prefix,
            limit=max_results,
        )

        return [
            Metric(
                key=item["key"],
                value=float(item["value"]),
                timestamp=item.get("timestamp", 0),
                step=item.get("step", 0),
            )
            for item in items
        ]

    def set_tag(self, run_id: str, tag: RunTag) -> None:
        """Set a tag on a run."""
        experiment_id = self._resolve_run_experiment(run_id)
        self._write_run_tag(experiment_id, run_id, tag)

    def delete_tag(self, run_id: str, key: str) -> None:
        """Delete a tag from a run."""
        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_RUN_PREFIX}{run_id}{SK_TAG_PREFIX}{key}"
        self._table.delete_item(pk=pk, sk=sk)

    def log_inputs(
        self,
        run_id: str,
        datasets: list[DatasetInput] | None = None,
        models: Any = None,
    ) -> None:
        raise MlflowException("Not implemented in Plan 1 — see Task 15")

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        raise MlflowException("Deferred to Plan 3")
