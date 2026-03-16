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
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
    ViewType,
)
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey

from mlflow_dynamodbstore.cache import ResolutionCache
from mlflow_dynamodbstore.dynamodb.config import ConfigReader
from mlflow_dynamodbstore.dynamodb.fts import (
    fts_diff,
    fts_items_for_text,
    tokenize_trigrams,
    tokenize_words,
)
from mlflow_dynamodbstore.dynamodb.keys import pad_step
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    CONFIG_TTL_POLICY,
    GSI1_CLIENT_PREFIX,
    GSI1_PK,
    GSI1_RUN_PREFIX,
    GSI1_SK,
    GSI1_TRACE_PREFIX,
    GSI2_EXPERIMENTS_PREFIX,
    GSI2_FTS_NAMES_PREFIX,
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
    PK_CONFIG,
    PK_EXPERIMENT_PREFIX,
    SK_DATASET_PREFIX,
    SK_DLINK_PREFIX,
    SK_EXPERIMENT_META,
    SK_EXPERIMENT_NAME_REV,
    SK_EXPERIMENT_TAG_PREFIX,
    SK_FTS_PREFIX,
    SK_FTS_REV_PREFIX,
    SK_INPUT_PREFIX,
    SK_INPUT_TAG_SUFFIX,
    SK_METRIC_HISTORY_PREFIX,
    SK_METRIC_PREFIX,
    SK_PARAM_PREFIX,
    SK_RANK_PREFIX,
    SK_RUN_PREFIX,
    SK_TAG_PREFIX,
    SK_TRACE_PREFIX,
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
        self._config = ConfigReader(self._table)
        self._config.reconcile()
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
            "tags": {},
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

        # Write NAME_REV item for suffix ILIKE support (GSI5)
        name_rev_item = {
            "PK": f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            "SK": SK_EXPERIMENT_NAME_REV,
            GSI5_PK: f"{GSI5_EXP_NAMES_PREFIX}{self._workspace}",
            GSI5_SK: f"REV#{_rev(name.lower())}#{exp_id}",
            "name": name,
        }
        self._table.put_item(name_rev_item)

        # Write tags if provided
        if tags:
            for tag in tags:
                self._write_experiment_tag(exp_id, tag)

        # Write FTS items for experiment name
        fts_items = fts_items_for_text(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            entity_type="E",
            entity_id=exp_id,
            field=None,
            text=name,
            workspace=self._workspace,
        )
        if fts_items:
            self._table.batch_write(fts_items)

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

        # Update NAME_REV item with new reversed name
        self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_NAME_REV,
            updates={
                GSI5_SK: f"REV#{_rev(new_name.lower())}#{experiment_id}",
                "name": new_name,
            },
        )

        # Update FTS items for experiment name change
        levels = ("W", "3")  # always trigram for experiment_name
        tokens_to_add, tokens_to_remove = fts_diff(old_name, new_name, levels)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        # Delete removed FTS items by looking up via reverse SK prefix
        if tokens_to_remove:
            rev_prefix = f"{SK_FTS_REV_PREFIX}E#{experiment_id}#"
            rev_items = self._table.query(pk=pk, sk_prefix=rev_prefix)
            for rev_item in rev_items:
                # Parse level and token from reverse SK: FTS_REV#E#<id>#<level>#<token>
                rev_sk = rev_item["SK"]
                # Build the corresponding forward SK to delete it too
                # rev_sk pattern: FTS_REV#E#<entity_id>#<level>#<token>
                parts = rev_sk[len(SK_FTS_REV_PREFIX) :].split("#")
                # parts = ["E", experiment_id, level, token]
                if len(parts) >= 4:
                    lvl, tok = parts[2], parts[3]
                    if (lvl, tok) in tokens_to_remove:
                        forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#E#{experiment_id}"
                        self._table.delete_item(pk=pk, sk=forward_sk)
                        self._table.delete_item(pk=pk, sk=rev_sk)

        # Write new FTS items for added tokens
        if tokens_to_add:
            new_fts_items: list[dict[str, Any]] = []
            for lvl, tok in tokens_to_add:
                forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#E#{experiment_id}"
                reverse_sk = f"{SK_FTS_REV_PREFIX}E#{experiment_id}#{lvl}#{tok}"
                gsi2pk_val = f"{GSI2_FTS_NAMES_PREFIX}{self._workspace}"
                gsi2sk_val = f"{lvl}#{tok}#E#{experiment_id}"
                new_fts_items.append(
                    {"PK": pk, "SK": forward_sk, GSI2_PK: gsi2pk_val, GSI2_SK: gsi2sk_val}
                )
                new_fts_items.append({"PK": pk, "SK": reverse_sk})
            self._table.batch_write(new_fts_items)

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
        """Search experiments with filter and order_by support."""
        from mlflow_dynamodbstore.dynamodb.search import parse_experiment_filter

        predicates = parse_experiment_filter(filter_string)

        # Classify predicates
        name_pred = next(
            (p for p in predicates if p.field_type == "attribute" and p.key == "name"),
            None,
        )
        tag_preds = [p for p in predicates if p.field_type == "tag"]

        if name_pred and name_pred.op == "=":
            experiments = self._search_experiments_by_name_exact(name_pred.value, view_type)
        elif name_pred and name_pred.op in ("LIKE", "ILIKE"):
            experiments = self._search_experiments_by_name_like(
                name_pred.value, name_pred.op, view_type
            )
        else:
            experiments = self._search_experiments_by_lifecycle(view_type, max_results)

        # Apply tag filters as post-filters
        if tag_preds:
            experiments = self._filter_experiments_by_tags(experiments, tag_preds)

        # Apply ordering
        if order_by:
            experiments = self._sort_experiments(experiments, order_by)

        return experiments[:max_results] if max_results else experiments

    # ------------------------------------------------------------------
    # search_experiments helpers
    # ------------------------------------------------------------------

    def _search_experiments_by_lifecycle(
        self, view_type: int, max_results: int
    ) -> list[Experiment]:
        """Query experiments by lifecycle stage using GSI2."""
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

        return experiments

    def _search_experiments_by_name_exact(self, name: str, view_type: int) -> list[Experiment]:
        """Exact name lookup via GSI3."""
        results = self._table.query(
            pk=f"{GSI3_EXP_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if not results:
            return []

        exp_id = results[0]["PK"].replace(PK_EXPERIMENT_PREFIX, "")
        exp = self.get_experiment(exp_id)

        # Filter by view_type
        if not self._matches_view_type(exp, view_type):
            return []

        return [exp]

    def _search_experiments_by_name_like(
        self, pattern: str, op: str, view_type: int
    ) -> list[Experiment]:
        """Search experiments by name pattern using GSI5 or FTS.

        Strategies:
        - 'prefix%': GSI5 begins_with on name
        - '%suffix': GSI5 reversed name begins_with
        - '%word%': GSI2 FTS index
        """
        is_ilike = op == "ILIKE"
        stripped = pattern

        has_leading = stripped.startswith("%")
        has_trailing = stripped.endswith("%")
        core = stripped.strip("%")

        if not has_leading and has_trailing:
            # Prefix match: name LIKE 'prod%'
            return self._search_experiments_by_name_prefix(core, view_type)
        elif has_leading and not has_trailing:
            # Suffix match: name ILIKE '%pipeline'
            return self._search_experiments_by_name_suffix(core, view_type, is_ilike)
        elif has_leading and has_trailing:
            # Contains match: name LIKE '%pipeline%'
            return self._search_experiments_by_name_contains(core, view_type, is_ilike)
        else:
            # No wildcards — treat as exact match
            return self._search_experiments_by_name_exact(core, view_type)

    def _search_experiments_by_name_prefix(self, prefix: str, view_type: int) -> list[Experiment]:
        """Use GSI5 to find experiments whose name starts with prefix."""
        items = self._table.query(
            pk=f"{GSI5_EXP_NAMES_PREFIX}{self._workspace}",
            sk_prefix=prefix,
            index_name="gsi5",
        )
        experiments: list[Experiment] = []
        for item in items:
            # GSI5 SK is "name#exp_id" — skip NAME_REV items (start with "REV#")
            gsi5sk = item.get(GSI5_SK, "")
            if gsi5sk.startswith("REV#"):
                continue
            exp_id = item["PK"].replace(PK_EXPERIMENT_PREFIX, "")
            exp = self.get_experiment(exp_id)
            if self._matches_view_type(exp, view_type):
                experiments.append(exp)
        return experiments

    def _search_experiments_by_name_suffix(
        self, suffix: str, view_type: int, case_insensitive: bool = True
    ) -> list[Experiment]:
        """Use GSI5 NAME_REV items to find experiments whose name ends with suffix."""
        reversed_suffix = _rev(suffix.lower())
        items = self._table.query(
            pk=f"{GSI5_EXP_NAMES_PREFIX}{self._workspace}",
            sk_prefix=f"REV#{reversed_suffix}",
            index_name="gsi5",
        )
        experiments: list[Experiment] = []
        for item in items:
            exp_id = item["PK"].replace(PK_EXPERIMENT_PREFIX, "")
            exp = self.get_experiment(exp_id)
            if self._matches_view_type(exp, view_type):
                experiments.append(exp)
        return experiments

    def _search_experiments_by_name_contains(
        self, substring: str, view_type: int, case_insensitive: bool = False
    ) -> list[Experiment]:
        """Use GSI2 FTS index to find experiments whose name contains substring."""
        # Try word tokens first, then trigrams
        word_tokens = tokenize_words(substring)
        exp_ids: set[str] | None = None

        if word_tokens:
            for token in word_tokens:
                fts_items = self._table.query(
                    pk=f"{GSI2_FTS_NAMES_PREFIX}{self._workspace}",
                    sk_prefix=f"W#{token}#E#",
                    index_name="gsi2",
                )
                ids = set()
                for item in fts_items:
                    gsi2sk = item.get(GSI2_SK, "")
                    # Pattern: W#<token>#E#<exp_id>
                    parts = gsi2sk.split("#")
                    for i, part in enumerate(parts):
                        if part == "E" and i + 1 < len(parts):
                            ids.add(parts[i + 1])
                            break
                exp_ids = ids if exp_ids is None else exp_ids & ids
        else:
            # Fallback to trigrams
            trigram_tokens = tokenize_trigrams(substring)
            for token in trigram_tokens:
                fts_items = self._table.query(
                    pk=f"{GSI2_FTS_NAMES_PREFIX}{self._workspace}",
                    sk_prefix=f"3#{token}#E#",
                    index_name="gsi2",
                )
                ids = set()
                for item in fts_items:
                    gsi2sk = item.get(GSI2_SK, "")
                    parts = gsi2sk.split("#")
                    for i, part in enumerate(parts):
                        if part == "E" and i + 1 < len(parts):
                            ids.add(parts[i + 1])
                            break
                exp_ids = ids if exp_ids is None else exp_ids & ids

        if not exp_ids:
            return []

        experiments: list[Experiment] = []
        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            if self._matches_view_type(exp, view_type):
                # Verify the substring actually appears in the name (FTS can have false positives)
                name = exp.name
                check_name = name.lower() if case_insensitive else name
                check_sub = substring.lower() if case_insensitive else substring
                if check_sub in check_name:
                    experiments.append(exp)
        return experiments

    def _filter_experiments_by_tags(
        self, experiments: list[Experiment], tag_preds: list[Any]
    ) -> list[Experiment]:
        """Post-filter experiments by tag predicates."""
        from mlflow_dynamodbstore.dynamodb.search import _compare

        filtered: list[Experiment] = []
        for exp in experiments:
            # Experiment.tags is a dict {key: value}
            tag_map = exp.tags if isinstance(exp.tags, dict) else {}
            if all(_compare(tag_map.get(pred.key), pred.op, pred.value) for pred in tag_preds):
                filtered.append(exp)
        return filtered

    def _sort_experiments(
        self, experiments: list[Experiment], order_by: list[str]
    ) -> list[Experiment]:
        """Sort experiments by the given order_by clauses (Python sort)."""
        for token in reversed(order_by):
            token = token.strip()
            parts = token.rsplit(None, 1)
            if len(parts) == 2 and parts[1].upper() in ("ASC", "DESC"):
                key_name = parts[0].strip()
                reverse = parts[1].upper() == "DESC"
            else:
                key_name = token
                reverse = False

            # Remove "attribute." prefix if present
            if "." in key_name:
                _, key_name = key_name.split(".", 1)

            def _sort_key(e: Experiment, k: str = key_name) -> str:
                val = getattr(e, k, None)
                return str(val) if val is not None else ""

            experiments = sorted(experiments, key=_sort_key, reverse=reverse)
        return experiments

    @staticmethod
    def _matches_view_type(exp: Experiment, view_type: int) -> bool:
        """Check if an experiment matches the requested view type."""
        if view_type == ViewType.ALL:
            return True
        if view_type == ViewType.ACTIVE_ONLY:
            return bool(exp.lifecycle_stage == "active")
        if view_type == ViewType.DELETED_ONLY:
            return bool(exp.lifecycle_stage == "deleted")
        return True

    # ------------------------------------------------------------------
    # Experiment tags
    # ------------------------------------------------------------------

    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        """Set a tag on an experiment."""
        # Verify experiment exists
        self.get_experiment(experiment_id)
        self._write_experiment_tag(experiment_id, tag)

    def _write_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        """Write an experiment tag item and optionally denormalize into the META item."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        item = {
            "PK": pk,
            "SK": f"{SK_EXPERIMENT_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(item)
        if self._config.should_denormalize(None, tag.key):
            self._denormalize_tag(pk, SK_EXPERIMENT_META, tag.key, tag.value)

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
            "tags": {},
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

        # Write FTS items for run name
        if run_name:
            run_fts = fts_items_for_text(
                pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
                entity_type="R",
                entity_id=run_id,
                field=None,
                text=run_name,
            )
            if run_fts:
                self._table.batch_write(run_fts)

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

        # Update FTS for run name if it changed
        old_run_name = current.get("run_name", "")
        if run_name and run_name != old_run_name:
            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            self._update_fts_for_rename(
                pk=pk,
                entity_type="R",
                entity_id=run_id,
                field=None,
                old_text=old_run_name or None,
                new_text=run_name,
                workspace=None,
            )

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
        """Search runs across experiments using parse -> plan -> execute pipeline."""
        from mlflow_dynamodbstore.dynamodb.pagination import (
            decode_page_token,
            encode_page_token,
        )
        from mlflow_dynamodbstore.dynamodb.search import (
            execute_query,
            parse_run_filter,
            plan_run_query,
        )

        # 1. Parse filter
        predicates = parse_run_filter(filter_string)

        # 2. Plan query
        denormalized_patterns = self._config.get_denormalize_patterns()
        plan = plan_run_query(predicates, order_by, run_view_type, denormalized_patterns)

        # 3. For each experiment: execute query
        # Handle multi-experiment pagination
        token_data = decode_page_token(page_token)
        exp_idx = token_data.get("exp_idx", 0) if token_data else 0
        inner_token = token_data.get("inner_token") if token_data else None

        runs: list[Run] = []
        remaining = max_results
        next_page_token: str | None = None

        for i, exp_id in enumerate(experiment_ids[exp_idx:], start=exp_idx):
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
            current_token = inner_token if i == exp_idx else None

            # Loop within experiment to handle non-run items in results
            inner_next: str | None = None
            while remaining > 0:
                items, inner_next = execute_query(
                    table=self._table,
                    plan=plan,
                    pk=pk,
                    max_results=remaining,
                    page_token=current_token,
                    predicates=predicates,
                )

                for item in items:
                    if "run_id" not in item:
                        continue
                    run_id = item["run_id"]
                    self._cache.put("run_exp", run_id, exp_id)

                    # Build full Run with tags/params/metrics
                    run_prefix = f"{SK_RUN_PREFIX}{run_id}"
                    tag_items = self._table.query(pk=pk, sk_prefix=f"{run_prefix}{SK_TAG_PREFIX}")
                    param_items = self._table.query(
                        pk=pk, sk_prefix=f"{run_prefix}{SK_PARAM_PREFIX}"
                    )
                    metric_items = self._table.query(
                        pk=pk, sk_prefix=f"{run_prefix}{SK_METRIC_PREFIX}"
                    )
                    runs.append(_item_to_run(item, tag_items, param_items, metric_items))
                    remaining -= 1

                    if remaining <= 0:
                        break

                # If no more pages in this experiment, move on
                if not inner_next:
                    break
                # If we still need more, fetch the next page
                current_token = inner_next

            if remaining <= 0:
                # Create pagination token
                if inner_next or i < len(experiment_ids) - 1:
                    next_page_token = encode_page_token(
                        {
                            "exp_idx": i if inner_next else i + 1,
                            "inner_token": inner_next,
                        }
                    )
                break

            if inner_next:
                next_page_token = encode_page_token(
                    {
                        "exp_idx": i,
                        "inner_token": inner_next,
                    }
                )
                break

        return runs, next_page_token

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
        """Write a run tag item and optionally denormalize into the META item."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        item = {
            "PK": pk,
            "SK": f"{SK_RUN_PREFIX}{run_id}{SK_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(item)
        if self._config.should_denormalize(experiment_id, tag.key):
            self._denormalize_tag(pk, f"{SK_RUN_PREFIX}{run_id}", tag.key, tag.value)
        # Write FTS items for tag value if configured
        if self._config.should_trigram("run_tag_value") and tag.value:
            tag_fts = fts_items_for_text(
                pk=pk,
                entity_type="R",
                entity_id=run_id,
                field=tag.key,
                text=tag.value,
            )
            if tag_fts:
                self._table.batch_write(tag_fts)

    def _build_run(self, item: dict[str, Any], tags: list[RunTag] | None = None) -> Run:
        """Build a Run entity from an item and optional in-memory tags."""
        info = _item_to_run_info(item)
        data = RunData(
            tags=tags or [],
            params=[],
            metrics=[],
        )
        return Run(run_info=info, run_data=data)

    def _denormalize_tag(self, pk: str, sk: str, tag_key: str, tag_value: str) -> None:
        """Write tag value into the META item's nested `tags` map."""
        self._table._table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression="SET #tags.#k = :v",
            ExpressionAttributeNames={"#tags": "tags", "#k": tag_key},
            ExpressionAttributeValues={":v": tag_value},
        )

    def _remove_denormalized_tag(self, pk: str, sk: str, tag_key: str) -> None:
        """Remove a tag from the META item's nested `tags` map."""
        self._table._table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression="REMOVE #tags.#k",
            ExpressionAttributeNames={"#tags": "tags", "#k": tag_key},
        )

    def _delete_fts_for_entity_field(
        self,
        pk: str,
        entity_type: str,
        entity_id: str,
        field: str | None,
    ) -> None:
        """Delete all FTS items for a given entity + field (e.g. when a tag is deleted)."""
        field_suffix = f"#{field}" if field else ""
        entity_prefix = f"{entity_type}#{entity_id}{field_suffix}"
        rev_prefix = f"{SK_FTS_REV_PREFIX}{entity_prefix}#"
        rev_items = self._table.query(pk=pk, sk_prefix=rev_prefix)
        for rev_item in rev_items:
            rev_sk = rev_item["SK"]
            suffix = rev_sk[len(SK_FTS_REV_PREFIX) + len(entity_prefix) + 1 :]
            parts = suffix.split("#", 1)
            if len(parts) == 2:
                lvl, tok = parts[0], parts[1]
                forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#{entity_prefix}"
                self._table.delete_item(pk=pk, sk=forward_sk)
            self._table.delete_item(pk=pk, sk=rev_sk)

    def _update_fts_for_rename(
        self,
        pk: str,
        entity_type: str,
        entity_id: str,
        field: str | None,
        old_text: str | None,
        new_text: str,
        workspace: str | None,
    ) -> None:
        """Compute FTS diff and apply: delete removed token items, write new token items."""
        levels = ("W", "3")
        tokens_to_add, tokens_to_remove = fts_diff(old_text, new_text, levels)
        field_suffix = f"#{field}" if field else ""
        entity_prefix = f"{entity_type}#{entity_id}{field_suffix}"

        # Delete removed FTS items (both forward and reverse)
        if tokens_to_remove:
            rev_prefix = f"{SK_FTS_REV_PREFIX}{entity_prefix}#"
            rev_items = self._table.query(pk=pk, sk_prefix=rev_prefix)
            for rev_item in rev_items:
                rev_sk = rev_item["SK"]
                # rev_sk: FTS_REV#<entity_type>#<entity_id>[#<field>]#<level>#<token>
                suffix = rev_sk[len(SK_FTS_REV_PREFIX) + len(entity_prefix) + 1 :]
                parts = suffix.split("#", 1)
                if len(parts) == 2:
                    lvl, tok = parts[0], parts[1]
                    if (lvl, tok) in tokens_to_remove:
                        forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#{entity_prefix}"
                        self._table.delete_item(pk=pk, sk=forward_sk)
                        self._table.delete_item(pk=pk, sk=rev_sk)

        # Write new FTS items for added tokens
        if tokens_to_add:
            add_gsi2 = entity_type in ("E", "M") and workspace is not None
            new_fts_items: list[dict[str, Any]] = []
            for lvl, tok in tokens_to_add:
                forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#{entity_prefix}"
                reverse_sk = f"{SK_FTS_REV_PREFIX}{entity_prefix}#{lvl}#{tok}"
                forward: dict[str, Any] = {"PK": pk, "SK": forward_sk}
                if add_gsi2:
                    forward[GSI2_PK] = f"{GSI2_FTS_NAMES_PREFIX}{workspace}"
                    forward[GSI2_SK] = f"{lvl}#{tok}#{entity_prefix}"
                new_fts_items.append(forward)
                new_fts_items.append({"PK": pk, "SK": reverse_sk})
            self._table.batch_write(new_fts_items)

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

        # Write FTS items for param values if configured
        if self._config.should_trigram("run_param_value"):
            fts_param_items: list[dict[str, Any]] = []
            for param in params:
                if param.value:
                    fts_param_items.extend(
                        fts_items_for_text(
                            pk=pk,
                            entity_type="R",
                            entity_id=run_id,
                            field=param.key,
                            text=param.value,
                        )
                    )
            if fts_param_items:
                self._table.batch_write(fts_param_items)

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
        if self._config.should_denormalize(experiment_id, key):
            self._remove_denormalized_tag(pk, f"{SK_RUN_PREFIX}{run_id}", key)
        # Remove FTS items for tag value if FTS was configured
        if self._config.should_trigram("run_tag_value"):
            self._delete_fts_for_entity_field(pk=pk, entity_type="R", entity_id=run_id, field=key)

    def log_inputs(
        self,
        run_id: str,
        datasets: list[DatasetInput] | None = None,
        models: Any = None,
    ) -> None:
        """Log dataset inputs for a run."""
        if not datasets:
            return

        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        items: list[dict[str, Any]] = []

        for dataset_input in datasets:
            ds = dataset_input.dataset
            ds_uuid = generate_ulid()

            # Dataset item: PK=EXP#<exp_id>, SK=D#<name>#<digest>
            items.append(
                {
                    "PK": pk,
                    "SK": f"{SK_DATASET_PREFIX}{ds.name}#{ds.digest}",
                    "name": ds.name,
                    "digest": ds.digest,
                    "source_type": ds.source_type,
                    "source": ds.source,
                    "schema": ds.schema,
                    "profile": ds.profile,
                }
            )

            # Input link item: PK=EXP#<exp_id>, SK=R#<run_id>#INPUT#<ds_uuid>
            items.append(
                {
                    "PK": pk,
                    "SK": f"{SK_RUN_PREFIX}{run_id}{SK_INPUT_PREFIX}{ds_uuid}",
                    "dataset_name": ds.name,
                    "dataset_digest": ds.digest,
                }
            )

            # Input tag items and extract context tag
            context: str | None = None
            for tag in dataset_input.tags:
                items.append(
                    {
                        "PK": pk,
                        "SK": (
                            f"{SK_RUN_PREFIX}{run_id}{SK_INPUT_PREFIX}{ds_uuid}"
                            f"{SK_INPUT_TAG_SUFFIX}{tag.key}"
                        ),
                        "key": tag.key,
                        "value": tag.value,
                    }
                )
                if tag.key == "mlflow.data.context":
                    context = tag.value

            # DLINK materialized item: PK=EXP#<exp_id>, SK=DLINK#<name>#<digest>#R#<run_id>
            dlink_item: dict[str, Any] = {
                "PK": pk,
                "SK": f"{SK_DLINK_PREFIX}{ds.name}#{ds.digest}#{SK_RUN_PREFIX}{run_id}",
                "dataset_name": ds.name,
                "dataset_digest": ds.digest,
                "run_id": run_id,
            }
            if context is not None:
                dlink_item["context"] = context
            items.append(dlink_item)

        self._table.batch_write(items)

    # ------------------------------------------------------------------
    # Trace CRUD
    # ------------------------------------------------------------------

    def _get_trace_ttl(self) -> int:
        """Compute TTL epoch from CONFIG#TTL_POLICY.trace_retention_days (default 30)."""
        item = self._table.get_item(pk=PK_CONFIG, sk=CONFIG_TTL_POLICY)
        days = 30
        if item and "trace_retention_days" in item:
            days = int(item["trace_retention_days"])
        return int(time.time()) + days * 86400

    def _resolve_trace_experiment(self, trace_id: str) -> str:
        """Resolve trace_id to experiment_id, using cache then GSI1."""
        cached = self._cache.get("trace_exp", trace_id)
        if cached:
            return cached

        results = self._table.query(
            pk=f"{GSI1_TRACE_PREFIX}{trace_id}",
            index_name="gsi1",
            limit=1,
        )
        if not results:
            raise MlflowException(
                f"Trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # GSI1 SK is EXP#<exp_id>
        gsi1sk: str = results[0][GSI1_SK]
        experiment_id = gsi1sk[len(PK_EXPERIMENT_PREFIX) :]
        self._cache.put("trace_exp", trace_id, experiment_id)
        return experiment_id

    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        """Create a trace in DynamoDB from a TraceInfo object."""
        mlflow_exp = trace_info.trace_location.mlflow_experiment
        if mlflow_exp is None:
            raise MlflowException(
                "TraceInfo must have an MLflow experiment location.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        experiment_id = mlflow_exp.experiment_id
        trace_id = trace_info.trace_id
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}"

        ttl = self._get_trace_ttl()

        # Extract trace name from trace_metadata
        trace_name = trace_info.trace_metadata.get(TraceTagKey.TRACE_NAME, "")
        execution_duration = trace_info.execution_duration or 0
        request_time = trace_info.request_time
        state_str = str(trace_info.state)

        # Build META item
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "trace_id": trace_id,
            "experiment_id": experiment_id,
            "request_time": request_time,
            "execution_duration": execution_duration,
            "state": state_str,
            "ttl": ttl,
            "tags": {},
            # LSI attributes
            LSI1_SK: request_time,
            LSI2_SK: request_time + execution_duration,
            LSI3_SK: f"{state_str}#{request_time}",
            LSI4_SK: trace_name.lower() if trace_name else "",
            LSI5_SK: execution_duration,
            # GSI1: reverse lookup trace_id -> experiment_id
            GSI1_PK: f"{GSI1_TRACE_PREFIX}{trace_id}",
            GSI1_SK: f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
        }

        if trace_info.client_request_id:
            item["client_request_id"] = trace_info.client_request_id

        self._table.put_item(item, condition="attribute_not_exists(PK)")

        # Cache trace_id -> experiment_id
        self._cache.put("trace_exp", trace_id, experiment_id)

        # Write CLIENTPTR item if client_request_id is present
        if trace_info.client_request_id:
            ptr_item: dict[str, Any] = {
                "PK": pk,
                "SK": f"{SK_TRACE_PREFIX}{trace_id}#CLIENTPTR",
                "ttl": ttl,
                GSI1_PK: f"{GSI1_CLIENT_PREFIX}{trace_info.client_request_id}",
                GSI1_SK: f"{GSI1_TRACE_PREFIX}{trace_id}",
            }
            self._table.put_item(ptr_item)

        # Write request metadata items
        if trace_info.trace_metadata:
            rmeta_items: list[dict[str, Any]] = []
            for key, value in trace_info.trace_metadata.items():
                rmeta_items.append(
                    {
                        "PK": pk,
                        "SK": f"{SK_TRACE_PREFIX}{trace_id}#RMETA#{key}",
                        "key": key,
                        "value": value,
                        "ttl": ttl,
                    }
                )
            if rmeta_items:
                self._table.batch_write(rmeta_items)

        # Write initial tag items + denormalization + FTS
        if trace_info.tags:
            for tag_key, tag_value in trace_info.tags.items():
                self._write_trace_tag(experiment_id, trace_id, tag_key, tag_value, ttl)

        return trace_info

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        """Fetch a trace by ID, reconstructing TraceInfo from DynamoDB items."""
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}"

        meta = self._table.get_item(pk=pk, sk=sk)
        if meta is None:
            raise MlflowException(
                f"Trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Read tags
        tag_items = self._table.query(
            pk=pk,
            sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#TAG#",
        )
        tags = {item["key"]: item["value"] for item in tag_items}

        # Read request metadata
        rmeta_items = self._table.query(
            pk=pk,
            sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#RMETA#",
        )
        trace_metadata = {item["key"]: item["value"] for item in rmeta_items}

        state_str = meta.get("state", "STATE_UNSPECIFIED")
        state = TraceState(state_str)

        return TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation(
                type=TraceLocationType.MLFLOW_EXPERIMENT,
                mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
            ),
            request_time=int(meta["request_time"]),
            execution_duration=int(meta.get("execution_duration", 0)),
            state=state,
            trace_metadata=trace_metadata,
            tags=tags,
            client_request_id=meta.get("client_request_id"),
        )

    def _write_trace_tag(
        self,
        experiment_id: str,
        trace_id: str,
        tag_key: str,
        tag_value: str,
        ttl: int,
    ) -> None:
        """Write a trace tag item with TTL, optionally denormalize, and write FTS items."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        item = {
            "PK": pk,
            "SK": f"{SK_TRACE_PREFIX}{trace_id}#TAG#{tag_key}",
            "key": tag_key,
            "value": tag_value,
            "ttl": ttl,
        }
        self._table.put_item(item)
        if self._config.should_denormalize(experiment_id, tag_key):
            self._denormalize_tag(pk, f"{SK_TRACE_PREFIX}{trace_id}", tag_key, tag_value)
        # Write FTS items for tag value if configured
        if self._config.should_trigram("trace_tag_value") and tag_value:
            tag_fts = fts_items_for_text(
                pk=pk,
                entity_type="T",
                entity_id=trace_id,
                field=tag_key,
                text=tag_value,
            )
            if tag_fts:
                self._table.batch_write(tag_fts)

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        """Set a tag on a trace."""
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        # Read TTL from the trace META item
        meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
        if meta is None:
            raise MlflowException(
                f"Trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        ttl = int(meta.get("ttl", self._get_trace_ttl()))
        self._write_trace_tag(experiment_id, trace_id, key, value, ttl)

    def delete_trace_tag(self, trace_id: str, key: str) -> None:
        """Delete a tag from a trace."""
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#TAG#{key}"
        self._table.delete_item(pk=pk, sk=sk)
        if self._config.should_denormalize(experiment_id, key):
            self._remove_denormalized_tag(pk, f"{SK_TRACE_PREFIX}{trace_id}", key)
        # Remove FTS items for tag value if FTS was configured
        if self._config.should_trigram("trace_tag_value"):
            self._delete_fts_for_entity_field(pk=pk, entity_type="T", entity_id=trace_id, field=key)

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        """Link traces to a run by writing mlflow.sourceRun request metadata."""
        for trace_id in trace_ids:
            experiment_id = self._resolve_trace_experiment(trace_id)
            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            # Read TTL from the trace META item
            meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
            ttl = int(meta["ttl"]) if meta and "ttl" in meta else self._get_trace_ttl()
            rmeta_item: dict[str, Any] = {
                "PK": pk,
                "SK": f"{SK_TRACE_PREFIX}{trace_id}#RMETA#{TraceMetadataKey.SOURCE_RUN}",
                "key": TraceMetadataKey.SOURCE_RUN,
                "value": run_id,
                "ttl": ttl,
            }
            self._table.put_item(rmeta_item)
