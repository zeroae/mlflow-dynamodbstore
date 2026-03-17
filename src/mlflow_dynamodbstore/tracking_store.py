"""DynamoDB-backed MLflow tracking store."""

from __future__ import annotations

import hashlib
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.entities.model_registry import PromptVersion
    from mlflow.entities.trace import Trace
    from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig

    from mlflow_dynamodbstore.xray.client import XRayClient

from mlflow.entities import (
    Assessment,
    Dataset,
    DatasetInput,
    EvaluationDataset,
    Experiment,
    ExperimentTag,
    InputTag,
    LoggedModel,
    LoggedModelParameter,
    LoggedModelTag,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunInputs,
    RunTag,
    ScorerVersion,
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
    ViewType,
)
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.protos.service_pb2 import DatasetSummary
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.utils.proto_json_utils import milliseconds_to_proto_timestamp
from mlflow.utils.uri import append_to_uri_path

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
    GSI1_CLIENT_PREFIX,
    GSI1_DS_EXP_PREFIX,
    GSI1_LM_PREFIX,
    GSI1_PK,
    GSI1_RUN_PREFIX,
    GSI1_SCOR_PREFIX,
    GSI1_SK,
    GSI1_TRACE_PREFIX,
    GSI2_ACTIVE_SCORERS_PREFIX,
    GSI2_DS_LIST_PREFIX,
    GSI2_EXPERIMENTS_PREFIX,
    GSI2_FTS_NAMES_PREFIX,
    GSI2_PK,
    GSI2_SESSIONS_PREFIX,
    GSI2_SK,
    GSI3_DS_NAME_PREFIX,
    GSI3_EXP_NAME_PREFIX,
    GSI3_PK,
    GSI3_SCOR_NAME_PREFIX,
    GSI3_SK,
    GSI5_EXP_NAMES_PREFIX,
    GSI5_PK,
    GSI5_SK,
    LSI1_SK,
    LSI2_SK,
    LSI3_SK,
    LSI4_SK,
    LSI5_SK,
    PK_DATASET_PREFIX,
    PK_EXPERIMENT_PREFIX,
    SK_DATASET_EXP_PREFIX,
    SK_DATASET_META,
    SK_DATASET_PREFIX,
    SK_DATASET_RECORD_PREFIX,
    SK_DATASET_TAG_PREFIX,
    SK_DLINK_PREFIX,
    SK_EXPERIMENT_META,
    SK_EXPERIMENT_NAME_REV,
    SK_EXPERIMENT_TAG_PREFIX,
    SK_FTS_PREFIX,
    SK_FTS_REV_PREFIX,
    SK_INPUT_PREFIX,
    SK_INPUT_TAG_SUFFIX,
    SK_LM_METRIC_PREFIX,
    SK_LM_PARAM_PREFIX,
    SK_LM_PREFIX,
    SK_LM_TAG_PREFIX,
    SK_METRIC_HISTORY_PREFIX,
    SK_METRIC_PREFIX,
    SK_PARAM_PREFIX,
    SK_RANK_LM_PREFIX,
    SK_RANK_LMD_PREFIX,
    SK_RANK_PREFIX,
    SK_RUN_PREFIX,
    SK_SCORER_OSCFG_SUFFIX,
    SK_SCORER_PREFIX,
    SK_SESSION_PREFIX,
    SK_TAG_PREFIX,
    SK_TRACE_PREFIX,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri
from mlflow_dynamodbstore.ids import generate_ulid, ulid_from_timestamp


def _rev(s: str) -> str:
    """Return reversed string."""
    return s[::-1]


class _ScorerVersionCompat(ScorerVersion):
    """ScorerVersion that handles non-integer experiment_id in to_proto().

    MLflow's ScorerVersion.to_proto() calls int(experiment_id) which fails
    for ULID string IDs. This subclass skips the experiment_id proto field
    since the handler's response already includes it as a string.
    """

    def to_proto(self) -> Any:
        from mlflow.protos.service_pb2 import Scorer as ProtoScorer

        proto = ProtoScorer()
        # Skip experiment_id — proto field is int32, our IDs are ULID strings.
        # The REST handler's response includes experiment_id separately.
        proto.scorer_name = self.scorer_name
        proto.scorer_version = self.scorer_version
        proto.serialized_scorer = self._serialized_scorer
        proto.creation_time = self.creation_time
        if self.scorer_id is not None:
            proto.scorer_id = self.scorer_id
        return proto


def _int_or_none(value: Any) -> int | None:
    """Convert a value to int, or return None if the value is None."""
    return int(value) if value is not None else None


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
        creation_time=_int_or_none(item.get("creation_time")),
        last_update_time=_int_or_none(item.get("last_update_time")),
    )


def _item_to_run_info(item: dict[str, Any]) -> RunInfo:
    """Convert a DynamoDB item to an MLflow RunInfo entity."""
    return RunInfo(
        run_id=item["run_id"],
        experiment_id=item["experiment_id"],
        user_id=item.get("user_id", ""),
        status=item.get("status", "RUNNING"),
        start_time=_int_or_none(item.get("start_time")),
        end_time=_int_or_none(item.get("end_time")),
        lifecycle_stage=item.get("lifecycle_stage", "active"),
        artifact_uri=item.get("artifact_uri", ""),
        run_name=item.get("run_name", ""),
    )


def _item_to_run(
    item: dict[str, Any],
    tags: list[dict[str, Any]],
    params: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    input_items: list[dict[str, Any]] | None = None,
    dataset_items: list[dict[str, Any]] | None = None,
    input_tag_items: list[dict[str, Any]] | None = None,
) -> Run:
    """Convert DynamoDB items to an MLflow Run entity."""
    info = _item_to_run_info(item)
    data = RunData(
        tags=[RunTag(t["key"], t["value"]) for t in tags],
        params=[Param(p["key"], p["value"]) for p in params],
        metrics=[
            Metric(m["key"], float(m["value"]), int(m.get("timestamp", 0)), int(m.get("step", 0)))
            for m in metrics
        ],
    )

    # Build RunInputs from input/dataset/input-tag items
    run_inputs = None
    if input_items:
        # Index datasets by name#digest
        ds_map: dict[str, dict[str, Any]] = {}
        for d in dataset_items or []:
            key = f"{d['name']}#{d['digest']}"
            ds_map[key] = d

        # Index input tags by the INPUT SK prefix they belong to
        itag_map: dict[str, list[dict[str, Any]]] = {}
        for t in input_tag_items or []:
            # SK format: R#<run_id>#INPUT#<uuid>#ITAG#<key>
            sk = t["SK"]
            itag_prefix = sk[: sk.index(SK_INPUT_TAG_SUFFIX)]
            itag_map.setdefault(itag_prefix, []).append(t)

        dataset_inputs = []
        for inp in input_items:
            ds_key = f"{inp['dataset_name']}#{inp['dataset_digest']}"
            ds_item = ds_map.get(ds_key)
            if ds_item is None:
                continue
            dataset = Dataset(
                name=ds_item["name"],
                digest=ds_item["digest"],
                source_type=ds_item.get("source_type", ""),
                source=ds_item.get("source", ""),
                schema=str(ds_item["schema"]) if ds_item.get("schema") is not None else None,
                profile=str(ds_item["profile"]) if ds_item.get("profile") is not None else None,
            )
            # The INPUT item SK is the prefix for its ITAG children
            inp_sk = inp["SK"]
            itags = [InputTag(key=t["key"], value=t["value"]) for t in itag_map.get(inp_sk, [])]
            dataset_inputs.append(DatasetInput(dataset=dataset, tags=itags))

        run_inputs = RunInputs(dataset_inputs=dataset_inputs)

    return Run(run_info=info, run_data=data, run_inputs=run_inputs)


def _item_to_logged_model(
    item: dict[str, Any],
    tags: list[dict[str, Any]] | None = None,
    params: list[dict[str, Any]] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> LoggedModel:
    """Convert DynamoDB items to a LoggedModel entity."""
    tag_dict = {t["key"]: t["value"] for t in (tags or [])}
    param_dict = {p["key"]: p["value"] for p in (params or [])}
    metric_list = [
        Metric(
            key=m["metric_name"],
            value=float(m["metric_value"]),
            timestamp=int(m.get("metric_timestamp_ms", 0)),
            step=int(m.get("metric_step", 0)),
        )
        for m in (metrics or [])
    ]

    # Merge denormalized tags/params from META item with sub-items
    meta_tags = item.get("tags", {})
    meta_params = item.get("params", {})
    for k, v in meta_tags.items():
        tag_dict.setdefault(k, v)
    for k, v in meta_params.items():
        param_dict.setdefault(k, v)

    return LoggedModel(
        experiment_id=item["experiment_id"],
        model_id=item["model_id"],
        name=item.get("name", ""),
        artifact_location=item.get("artifact_location", ""),
        creation_timestamp=int(item.get("creation_timestamp_ms", 0)),
        last_updated_timestamp=int(item.get("last_updated_timestamp_ms", 0)),
        model_type=item.get("model_type") or None,
        source_run_id=item.get("source_run_id") or None,
        status=LoggedModelStatus(item.get("status", "READY")),
        status_message=item.get("status_message") or None,
        tags=tag_dict,
        params=param_dict,
        metrics=metric_list,
    )


class DynamoDBTrackingStore(AbstractStore):
    """MLflow tracking store backed by DynamoDB."""

    def __init__(
        self,
        store_uri: str,
        artifact_uri: str | None = None,
    ) -> None:
        uri = parse_dynamodb_uri(store_uri)
        if uri.deploy:
            ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)
        self._uri = uri
        self._cache = ResolutionCache()
        self._artifact_uri = artifact_uri or "./mlartifacts"
        self.artifact_root_uri = self._artifact_uri
        self._workspace = "default"
        self._config = ConfigReader(self._table)
        self._config.reconcile()
        super().__init__()

    @property
    def supports_workspaces(self) -> bool:
        """DynamoDB store always supports workspaces.

        Workspace scoping is built into the schema (GSI2/GSI3 prefixes,
        META workspace attribute). The --enable-workspaces server flag
        controls whether workspace features are active at runtime.
        """
        return True

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
            LSI2_SK: now_ms,
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
                LSI2_SK: now_ms,
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
        """Soft-delete an experiment and set TTL on META only."""
        now_ms = int(time.time() * 1000)

        updates: dict[str, Any] = {
            "lifecycle_stage": "deleted",
            "last_update_time": now_ms,
            LSI1_SK: f"deleted#{experiment_id}",
            LSI2_SK: now_ms,
            GSI2_PK: f"{GSI2_EXPERIMENTS_PREFIX}{self._workspace}#deleted",
        }

        # Compute TTL if enabled
        ttl_seconds = self._config.get_soft_deleted_ttl_seconds()
        if ttl_seconds is not None:
            updates["ttl"] = int(time.time()) + ttl_seconds

        self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_META,
            updates=updates,
        )

    def restore_experiment(self, experiment_id: str) -> None:
        """Restore a soft-deleted experiment and remove TTL from META."""
        now_ms = int(time.time() * 1000)

        self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_META,
            updates={
                "lifecycle_stage": "active",
                "last_update_time": now_ms,
                LSI1_SK: f"active#{experiment_id}",
                LSI2_SK: now_ms,
                GSI2_PK: f"{GSI2_EXPERIMENTS_PREFIX}{self._workspace}#active",
            },
            removes=["ttl"],
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

        from mlflow.store.entities import PagedList

        results = experiments[:max_results] if max_results else experiments
        return PagedList(results, token=None)

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

    def delete_experiment_tag(self, experiment_id: str, key: str) -> None:
        """Delete a tag from an experiment."""
        self.get_experiment(experiment_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_EXPERIMENT_TAG_PREFIX}{key}"
        self._table.delete_item(pk=pk, sk=sk)
        if self._config.should_denormalize(None, key):
            self._remove_denormalized_tag(pk, SK_EXPERIMENT_META, key)

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
            # GSI1: reverse lookup run_id -> experiment_id
            GSI1_PK: f"{GSI1_RUN_PREFIX}{run_id}",
            GSI1_SK: f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
        }

        # LSI4 is sparse — only set when run_name is non-empty (DynamoDB rejects empty string keys)
        if run_name:
            item[LSI4_SK] = run_name.lower()

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

        # Query input items (INPUT links and their ITAG children share the same prefix)
        all_input_items = self._table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk_prefix=f"{run_prefix}{SK_INPUT_PREFIX}",
        )
        # Separate INPUT link items from ITAG items
        input_items = []
        input_tag_items = []
        for it in all_input_items:
            if SK_INPUT_TAG_SUFFIX in it["SK"]:
                input_tag_items.append(it)
            else:
                input_items.append(it)

        # Query dataset items for this experiment
        dataset_items = []
        if input_items:
            dataset_items = self._table.query(
                pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
                sk_prefix=SK_DATASET_PREFIX,
            )

        return _item_to_run(
            item,
            tag_items,
            param_items,
            metric_items,
            input_items=input_items,
            dataset_items=dataset_items,
            input_tag_items=input_tag_items,
        )

    # ------------------------------------------------------------------
    # Logged Model CRUD
    # ------------------------------------------------------------------

    def _resolve_logged_model_experiment(self, model_id: str) -> str:
        """Resolve experiment_id for a logged model via GSI1."""
        results = self._table.query(
            pk=f"{GSI1_LM_PREFIX}{model_id}",
            index_name="gsi1",
            limit=1,
        )
        if not results:
            raise MlflowException(
                f"LoggedModel '{model_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        gsi1sk: str = results[0][GSI1_SK]
        return gsi1sk.replace(PK_EXPERIMENT_PREFIX, "")

    def create_logged_model(
        self,
        experiment_id: str,
        name: str | None = None,
        source_run_id: str | None = None,
        tags: list[LoggedModelTag] | None = None,
        params: list[LoggedModelParameter] | None = None,
        model_type: str | None = None,
    ) -> LoggedModel:
        """Create a new logged model within an experiment."""
        exp = self.get_experiment(experiment_id)

        now_ms = int(time.time() * 1000)
        model_id = f"m-{generate_ulid()}"
        name = name or model_id
        artifact_location = f"{exp.artifact_location}/models/{model_id}/artifacts/"

        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_LM_PREFIX}{model_id}"

        tag_dict = {t.key: t.value for t in (tags or [])}
        param_dict = {p.key: p.value for p in (params or [])}

        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "model_id": model_id,
            "experiment_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location,
            "creation_timestamp_ms": now_ms,
            "last_updated_timestamp_ms": now_ms,
            "status": str(LoggedModelStatus.PENDING),
            "lifecycle_stage": "active",
            "model_type": model_type or "",
            "source_run_id": source_run_id or "",
            "status_message": "",
            "tags": tag_dict,
            "params": param_dict,
            "workspace": self._workspace,
            # LSI projections for filtering/sorting
            LSI1_SK: f"active#{model_id}",
            LSI2_SK: now_ms,
            LSI3_SK: f"PENDING#{model_id}",
            LSI4_SK: name.lower(),
            # GSI1: reverse lookup model_id -> experiment_id
            GSI1_PK: f"{GSI1_LM_PREFIX}{model_id}",
            GSI1_SK: f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
        }

        self._table.put_item(item, condition="attribute_not_exists(SK)")

        # Write tag sub-items
        for key, value in tag_dict.items():
            self._table.put_item(
                {
                    "PK": pk,
                    "SK": f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}{key}",
                    "key": key,
                    "value": value,
                }
            )

        # Write param sub-items
        for key, value in param_dict.items():
            self._table.put_item(
                {
                    "PK": pk,
                    "SK": f"{SK_LM_PREFIX}{model_id}{SK_LM_PARAM_PREFIX}{key}",
                    "key": key,
                    "value": value,
                }
            )

        return _item_to_logged_model(item)

    def get_logged_model(self, model_id: str, allow_deleted: bool = False) -> LoggedModel:
        """Fetch a logged model by ID."""
        experiment_id = self._resolve_logged_model_experiment(model_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_LM_PREFIX}{model_id}"

        meta = self._table.get_item(pk=pk, sk=sk)
        if meta is None:
            raise MlflowException(
                f"LoggedModel '{model_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        if meta.get("lifecycle_stage") == "deleted" and not allow_deleted:
            raise MlflowException(
                f"LoggedModel '{model_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Load sub-items
        tag_items = self._table.query(
            pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}"
        )
        param_items = self._table.query(
            pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_PARAM_PREFIX}"
        )
        metric_items = self._table.query(
            pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_METRIC_PREFIX}"
        )

        return _item_to_logged_model(meta, tag_items, param_items, metric_items)

    def finalize_logged_model(self, model_id: str, status: LoggedModelStatus) -> LoggedModel:
        """Update a logged model's status (e.g. READY or FAILED)."""
        experiment_id = self._resolve_logged_model_experiment(model_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_LM_PREFIX}{model_id}"
        now_ms = int(time.time() * 1000)

        self._table.update_item(
            pk=pk,
            sk=sk,
            updates={
                "status": str(status),
                "last_updated_timestamp_ms": now_ms,
                LSI3_SK: f"{status}#{model_id}",
            },
        )
        return self.get_logged_model(model_id)

    def delete_logged_model(self, model_id: str) -> None:
        """Soft-delete a logged model and set TTL on all related items."""
        experiment_id = self._resolve_logged_model_experiment(model_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_LM_PREFIX}{model_id}"
        now_ms = int(time.time() * 1000)

        ttl_seconds = self._config.get_soft_deleted_ttl_seconds()
        ttl_value = int(time.time()) + ttl_seconds if ttl_seconds is not None else None

        updates: dict[str, Any] = {
            "lifecycle_stage": "deleted",
            "last_updated_timestamp_ms": now_ms,
            LSI1_SK: f"deleted#{model_id}",
        }
        if ttl_value is not None:
            updates["ttl"] = ttl_value
        self._table.update_item(pk=pk, sk=sk, updates=updates)

        # Set TTL on child items (tags, params, metrics)
        children = self._table.query(pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}#")
        for child in children:
            if ttl_value is not None:
                self._table.update_item(pk=pk, sk=child["SK"], updates={"ttl": ttl_value})

        # Set TTL on RANK items built from metric sub-items
        metric_children = [c for c in children if SK_LM_METRIC_PREFIX in c["SK"]]
        for mc in metric_children:
            inv_value = self._invert_metric_value(float(mc["metric_value"]))
            rank_sk = f"{SK_RANK_LM_PREFIX}{mc['metric_name']}#{inv_value}#{model_id}"
            if ttl_value is not None:
                self._table.update_item(pk=pk, sk=rank_sk, updates={"ttl": ttl_value})
            if mc.get("dataset_name") and mc.get("dataset_digest"):
                rank_sk_ds = (
                    f"{SK_RANK_LMD_PREFIX}{mc['metric_name']}#"
                    f"{mc['dataset_name']}#{mc['dataset_digest']}#{inv_value}#{model_id}"
                )
                if ttl_value is not None:
                    self._table.update_item(pk=pk, sk=rank_sk_ds, updates={"ttl": ttl_value})

    def set_logged_model_tags(self, model_id: str, tags: list[LoggedModelTag]) -> None:
        """Set (or overwrite) tags on a logged model."""
        experiment_id = self._resolve_logged_model_experiment(model_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        now_ms = int(time.time() * 1000)

        tag_dict: dict[str, str] = {}
        for tag in tags:
            self._table.put_item(
                {
                    "PK": pk,
                    "SK": f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}{tag.key}",
                    "key": tag.key,
                    "value": tag.value,
                }
            )
            tag_dict[tag.key] = tag.value

        # Update denormalized tags on META item
        meta = self._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model_id}") or {}
        existing_tags: dict[str, str] = meta.get("tags", {})
        existing_tags.update(tag_dict)
        self._table.update_item(
            pk=pk,
            sk=f"{SK_LM_PREFIX}{model_id}",
            updates={"tags": existing_tags, "last_updated_timestamp_ms": now_ms},
        )

    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        """Delete a tag from a logged model."""
        experiment_id = self._resolve_logged_model_experiment(model_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        now_ms = int(time.time() * 1000)

        self._table.delete_item(pk=pk, sk=f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}{key}")

        # Update denormalized tags on META item
        meta = self._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model_id}") or {}
        existing_tags: dict[str, str] = meta.get("tags", {})
        existing_tags.pop(key, None)
        self._table.update_item(
            pk=pk,
            sk=f"{SK_LM_PREFIX}{model_id}",
            updates={"tags": existing_tags, "last_updated_timestamp_ms": now_ms},
        )

    def record_logged_model(self, run_id: str, mlflow_model: dict[str, Any]) -> None:
        """Append a model info dict to the run's mlflow.loggedModels tag."""
        import json as _json

        run = self.get_run(run_id)
        experiment_id = run.info.experiment_id
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        tag_sk = f"{SK_RUN_PREFIX}{run_id}{SK_TAG_PREFIX}mlflow.loggedModels"
        existing = self._table.get_item(pk=pk, sk=tag_sk)
        models = _json.loads(existing["value"]) if existing else []
        models.append(mlflow_model)
        serialized = _json.dumps(models)

        self._table.put_item(
            {
                "PK": pk,
                "SK": tag_sk,
                "key": "mlflow.loggedModels",
                "value": serialized,
            }
        )

        # Update denormalized tags on run META
        run_sk = f"{SK_RUN_PREFIX}{run_id}"
        meta = self._table.get_item(pk=pk, sk=run_sk)
        run_tags = meta.get("tags", {}) if meta else {}
        run_tags["mlflow.loggedModels"] = serialized
        self._table.update_item(pk=pk, sk=run_sk, updates={"tags": run_tags})

    def search_logged_models(
        self,
        experiment_ids: list[str],
        filter_string: str | None = None,
        datasets: list[dict[str, Any]] | None = None,
        max_results: int | None = None,
        order_by: list[dict[str, Any]] | None = None,
        page_token: str | None = None,
    ) -> PagedList[LoggedModel]:
        """Search logged models across experiments using parse/plan/execute pipeline.

        Collects all matching items from every experiment first, then paginates
        the merged result using an offset-based token.
        """
        from mlflow_dynamodbstore.dynamodb.pagination import decode_page_token, encode_page_token
        from mlflow_dynamodbstore.dynamodb.search import (
            execute_logged_model_query,
            parse_logged_model_filter,
            plan_logged_model_query,
        )

        max_results = max_results or 100
        predicates = parse_logged_model_filter(filter_string)
        plan = plan_logged_model_query(predicates, order_by, datasets)

        # Collect all matching items across experiments (pagination applied later)
        all_items: list[dict[str, Any]] = []
        for exp_id in experiment_ids:
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
            items = execute_logged_model_query(self._table, plan, pk, predicates)
            all_items.extend(items)

        # Convert items to LoggedModel entities
        models: list[LoggedModel] = []
        for item in all_items:
            exp_id = item["experiment_id"]
            model_id = item["model_id"]
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
            tag_items = self._table.query(
                pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}"
            )
            param_items = self._table.query(
                pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_PARAM_PREFIX}"
            )
            metric_items = self._table.query(
                pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_METRIC_PREFIX}"
            )
            models.append(_item_to_logged_model(item, tag_items, param_items, metric_items))

        # Apply offset-based pagination across the merged result
        token_data = decode_page_token(page_token)
        offset = token_data.get("offset", 0) if token_data else 0
        page = models[offset : offset + max_results]
        has_more = len(models) > offset + max_results
        next_token = encode_page_token({"offset": offset + max_results}) if has_more else None

        return PagedList(page, next_token)

    @staticmethod
    def _invert_metric_value(value: float) -> str:
        """Invert a metric value for descending sort in DynamoDB RANK items."""
        max_val = 9999999999.9999
        inv = max_val - value
        return f"{inv:020.4f}"

    def _log_logged_model_metric(
        self,
        experiment_id: str,
        model_id: str,
        metric_name: str,
        metric_value: float,
        metric_timestamp_ms: int,
        metric_step: int,
        run_id: str,
        dataset_name: str | None = None,
        dataset_digest: str | None = None,
    ) -> None:
        """Write a metric sub-item and RANK items for a logged model."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        metric_sk = f"{SK_LM_PREFIX}{model_id}{SK_LM_METRIC_PREFIX}{metric_name}#{run_id}"

        # Check for existing metric to delete old RANK item(s) first
        existing = self._table.get_item(pk=pk, sk=metric_sk)
        if existing is not None:
            old_inv = self._invert_metric_value(float(existing["metric_value"]))
            old_rank_sk = f"{SK_RANK_LM_PREFIX}{metric_name}#{old_inv}#{model_id}"
            self._table.delete_item(pk=pk, sk=old_rank_sk)
            if existing.get("dataset_name") and existing.get("dataset_digest"):
                old_rank_ds = (
                    f"{SK_RANK_LMD_PREFIX}{metric_name}#"
                    f"{existing['dataset_name']}#{existing['dataset_digest']}#{old_inv}#{model_id}"
                )
                self._table.delete_item(pk=pk, sk=old_rank_ds)

        # Write metric sub-item
        metric_item: dict[str, Any] = {
            "PK": pk,
            "SK": metric_sk,
            "metric_name": metric_name,
            "metric_value": Decimal(str(metric_value)),
            "metric_timestamp_ms": metric_timestamp_ms,
            "metric_step": metric_step,
            "run_id": run_id,
            "model_id": model_id,
        }
        if dataset_name:
            metric_item["dataset_name"] = dataset_name
        if dataset_digest:
            metric_item["dataset_digest"] = dataset_digest
        self._table.put_item(metric_item)

        # Write global RANK item (descending by inverted value)
        inv_value = self._invert_metric_value(metric_value)
        self._table.put_item(
            {
                "PK": pk,
                "SK": f"{SK_RANK_LM_PREFIX}{metric_name}#{inv_value}#{model_id}",
                "model_id": model_id,
                "metric_name": metric_name,
                "metric_value": Decimal(str(metric_value)),
            }
        )

        # Write dataset-scoped RANK item
        if dataset_name and dataset_digest:
            self._table.put_item(
                {
                    "PK": pk,
                    "SK": (
                        f"{SK_RANK_LMD_PREFIX}{metric_name}#"
                        f"{dataset_name}#{dataset_digest}#{inv_value}#{model_id}"
                    ),
                    "model_id": model_id,
                    "metric_name": metric_name,
                    "metric_value": Decimal(str(metric_value)),
                    "dataset_name": dataset_name,
                    "dataset_digest": dataset_digest,
                }
            )

        # Update last_updated_timestamp_ms on the model meta item
        now_ms = int(time.time() * 1000)
        self._table.update_item(
            pk=pk,
            sk=f"{SK_LM_PREFIX}{model_id}",
            updates={"last_updated_timestamp_ms": now_ms},
        )

    def update_run_info(
        self, run_id: str, run_status: str | int, end_time: int, run_name: str
    ) -> RunInfo:
        """Update run status, end_time, and run_name."""
        from mlflow.entities import RunStatus

        # MLflow may pass status as protobuf enum int (e.g. 3 for FINISHED)
        if isinstance(run_status, int):
            run_status = RunStatus.to_string(run_status)

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
        }
        removes: list[str] = []
        if run_name:
            updates[LSI4_SK] = run_name.lower()
        else:
            removes.append(LSI4_SK)

        if end_time is not None:
            updates["end_time"] = end_time
            updates[LSI2_SK] = end_time
            start_time = current.get("start_time", 0)
            if start_time:
                updates[LSI5_SK] = str(end_time - start_time)

        updated_item = self._table.update_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=f"{SK_RUN_PREFIX}{run_id}",
            updates=updates,
            removes=removes if removes else None,
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
        """Soft-delete a run and set TTL on all related items."""
        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        # Compute TTL if enabled
        ttl_seconds = self._config.get_soft_deleted_ttl_seconds()
        ttl_value = int(time.time()) + ttl_seconds if ttl_seconds is not None else None

        # Update run META item
        updates: dict[str, Any] = {
            "lifecycle_stage": "deleted",
            LSI1_SK: f"deleted#{run_id}",
        }
        if ttl_value is not None:
            updates["ttl"] = ttl_value

        self._table.update_item(pk=pk, sk=f"{SK_RUN_PREFIX}{run_id}", updates=updates)

        # Set TTL on related items if enabled
        if ttl_value is not None:
            self._set_ttl_on_run_related_items(pk, run_id, ttl_value)

    def _set_ttl_on_run_related_items(self, pk: str, run_id: str, ttl_value: int) -> None:
        """Set TTL attribute on all items related to a run."""
        # 1. Children: SK begins_with R#<run_id>#
        children = self._table.query(pk=pk, sk_prefix=f"{SK_RUN_PREFIX}{run_id}#")
        for child in children:
            self._table.update_item(pk=pk, sk=child["SK"], updates={"ttl": ttl_value})

        # 2. RANK items containing run_id
        rank_items = self._table.query(pk=pk, sk_prefix=SK_RANK_PREFIX)
        for item in rank_items:
            if item.get("run_id") == run_id:
                self._table.update_item(pk=pk, sk=item["SK"], updates={"ttl": ttl_value})

        # 3. FTS_REV items: SK begins_with FTS_REV#R#<run_id>#
        fts_rev_items = self._table.query(
            pk=pk, sk_prefix=f"{SK_FTS_REV_PREFIX}{SK_RUN_PREFIX}{run_id}#"
        )
        for item in fts_rev_items:
            self._table.update_item(pk=pk, sk=item["SK"], updates={"ttl": ttl_value})
            # Also update the corresponding forward FTS item
            # FTS_REV SK pattern: FTS_REV#R#<run_id>#<fts_sk_suffix>
            # The forward FTS SK contains the run_id
            fwd_sk = item.get("fwd_sk")
            if fwd_sk:
                self._table.update_item(pk=pk, sk=fwd_sk, updates={"ttl": ttl_value})

        # 4. FTS items that contain the run_id in SK (catch any missed by FTS_REV)
        fts_items = self._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        for item in fts_items:
            if run_id in item.get("SK", ""):
                self._table.update_item(pk=pk, sk=item["SK"], updates={"ttl": ttl_value})

    def restore_run(self, run_id: str) -> None:
        """Restore a soft-deleted run and remove TTL from all related items."""
        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        # Update run META: restore lifecycle and remove TTL
        self._table.update_item(
            pk=pk,
            sk=f"{SK_RUN_PREFIX}{run_id}",
            updates={
                "lifecycle_stage": "active",
                LSI1_SK: f"active#{run_id}",
            },
            removes=["ttl"],
        )

        # Remove TTL from all related items
        self._remove_ttl_from_run_related_items(pk, run_id)

    def _remove_ttl_from_run_related_items(self, pk: str, run_id: str) -> None:
        """Remove TTL attribute from all items related to a run."""
        # 1. Children: SK begins_with R#<run_id>#
        children = self._table.query(pk=pk, sk_prefix=f"{SK_RUN_PREFIX}{run_id}#")
        for child in children:
            self._table.update_item(pk=pk, sk=child["SK"], removes=["ttl"])

        # 2. RANK items containing run_id
        rank_items = self._table.query(pk=pk, sk_prefix=SK_RANK_PREFIX)
        for item in rank_items:
            if item.get("run_id") == run_id:
                self._table.update_item(pk=pk, sk=item["SK"], removes=["ttl"])

        # 3. FTS_REV items: SK begins_with FTS_REV#R#<run_id>#
        fts_rev_items = self._table.query(
            pk=pk, sk_prefix=f"{SK_FTS_REV_PREFIX}{SK_RUN_PREFIX}{run_id}#"
        )
        for item in fts_rev_items:
            self._table.update_item(pk=pk, sk=item["SK"], removes=["ttl"])
            fwd_sk = item.get("fwd_sk")
            if fwd_sk:
                self._table.update_item(pk=pk, sk=fwd_sk, removes=["ttl"])

        # 4. FTS items that contain the run_id in SK
        fts_items = self._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        for item in fts_items:
            if run_id in item.get("SK", ""):
                self._table.update_item(pk=pk, sk=item["SK"], removes=["ttl"])

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

    def _upsert_session_tracker(
        self,
        experiment_id: str,
        session_id: str,
        timestamp_ms: int,
        ttl: int | None,
    ) -> None:
        """Upsert a session tracker item using atomic ADD + conditional SET."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_SESSION_PREFIX}{session_id}"
        gsi2pk = f"{GSI2_SESSIONS_PREFIX}{self._workspace}#{experiment_id}"

        # GSI2 SK must be a zero-padded string (GSI key schema is type S)
        gsi2sk_str = f"{timestamp_ms:020d}"

        # Combined ADD + SET requires raw boto3 (same pattern as _denormalize_tag)
        update_expr = (
            "ADD trace_count :one "
            "SET first_trace_timestamp_ms = if_not_exists(first_trace_timestamp_ms, :ts), "
            "last_trace_timestamp_ms = :ts, "
            "session_id = if_not_exists(session_id, :sid), "
            "#gsi2pk = :gsi2pk, #gsi2sk = :gsi2sk"
        )
        expr_names = {"#gsi2pk": "gsi2pk", "#gsi2sk": "gsi2sk"}
        expr_values: dict[str, Any] = {
            ":one": 1,
            ":ts": timestamp_ms,
            ":sid": session_id,
            ":gsi2pk": gsi2pk,
            ":gsi2sk": gsi2sk_str,
        }
        if ttl is not None:
            update_expr += ", #ttl = :ttl"
            expr_names["#ttl"] = "ttl"
            expr_values[":ttl"] = ttl

        self._table._table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
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
            hist_item: dict[str, Any] = {
                "PK": pk,
                "SK": hist_sk,
                "key": metric.key,
                "value": ddb_value,
                "timestamp": metric.timestamp,
                "step": metric.step,
            }
            metric_history_ttl = self._config.get_metric_history_ttl_seconds()
            if metric_history_ttl is not None:
                hist_item["ttl"] = int(time.time()) + metric_history_ttl
            items.append(hist_item)

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

        from mlflow.store.entities import PagedList

        metrics = [
            Metric(
                key=item["key"],
                value=float(item["value"]),
                timestamp=int(item.get("timestamp", 0)),
                step=int(item.get("step", 0)),
            )
            for item in items
        ]
        return PagedList(metrics, token=None)

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

    def _search_datasets(self, experiment_ids: list[str]) -> list[DatasetSummary]:
        """Search for legacy V2 datasets (D# and DLINK# items) under experiment partitions.

        This method queries existing D# (dataset) and DLINK# (dataset-run link) items
        that are stored under experiment partitions when log_inputs is called.

        Args:
            experiment_ids: List of experiment IDs to search within.

        Returns:
            List of DatasetSummary protobuf objects.
        """
        dataset_map: dict[tuple[str, str], dict[str, Any]] = {}

        for experiment_id in experiment_ids:
            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

            # Query D# items (dataset items)
            d_items = self._table.query(
                pk=pk,
                sk_prefix=SK_DATASET_PREFIX,
            )
            for item in d_items:
                # SK format: D#<name>#<digest>
                sk = item.get("SK", "")
                if sk.startswith(SK_DATASET_PREFIX):
                    parts = sk[len(SK_DATASET_PREFIX) :].split("#", 1)
                    if len(parts) == 2:
                        name, digest = parts
                        key = (name, digest)
                        if key not in dataset_map:
                            dataset_map[key] = {
                                "experiment_id": experiment_id,
                                "name": item.get("name", name),
                                "digest": item.get("digest", digest),
                                "context": None,
                            }

            # Query DLINK# items (dataset-run link items) for context
            dlink_items = self._table.query(
                pk=pk,
                sk_prefix=SK_DLINK_PREFIX,
            )
            for item in dlink_items:
                # Extract context if present (mlflow.data.context tag from log_inputs)
                if "context" in item:
                    # SK format: DLINK#<name>#<digest>#R#<run_id>
                    sk = item.get("SK", "")
                    if sk.startswith(SK_DLINK_PREFIX):
                        parts = sk[len(SK_DLINK_PREFIX) :].split("#", 2)
                        if len(parts) >= 2:
                            name, digest = parts[0], parts[1]
                            key = (name, digest)
                            if key in dataset_map:
                                dataset_map[key]["context"] = item["context"]

        # Build DatasetSummary protobuf objects
        results: list[DatasetSummary] = []
        for (name, digest), data in dataset_map.items():
            summary = DatasetSummary()
            exp_id: str = data["experiment_id"]
            ds_name: str = data["name"]
            ds_digest: str = data["digest"]
            summary.experiment_id = exp_id
            summary.name = ds_name
            summary.digest = ds_digest
            context = data["context"]
            if context:
                summary.context = context
            results.append(summary)

        return results

    # ------------------------------------------------------------------
    # Trace CRUD
    # ------------------------------------------------------------------

    def _get_trace_ttl(self) -> int | None:
        """Compute TTL epoch from CONFIG#TTL_POLICY.trace_retention_days (default 30).

        Returns None when trace_retention_days is 0 (TTL disabled).
        """
        days = self._config.get_ttl_policy()["trace_retention_days"]
        if days == 0:
            return None
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

        # Extract trace name from trace_metadata or tags
        trace_name = trace_info.trace_metadata.get(
            TraceTagKey.TRACE_NAME, ""
        ) or trace_info.tags.get(TraceTagKey.TRACE_NAME, "")
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
            "tags": {},
            # LSI attributes (must be strings, zero-padded for sort order)
            LSI1_SK: f"{request_time:020d}",
            LSI2_SK: request_time + execution_duration,
            LSI3_SK: f"{state_str}#{request_time:020d}",
            LSI5_SK: f"{execution_duration:020d}",
            # GSI1: reverse lookup trace_id -> experiment_id
            GSI1_PK: f"{GSI1_TRACE_PREFIX}{trace_id}",
            GSI1_SK: f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
        }

        if trace_name:
            item[LSI4_SK] = trace_name.lower()

        if ttl is not None:
            item["ttl"] = ttl

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
                GSI1_PK: f"{GSI1_CLIENT_PREFIX}{trace_info.client_request_id}",
                GSI1_SK: f"{GSI1_TRACE_PREFIX}{trace_id}",
            }
            if ttl is not None:
                ptr_item["ttl"] = ttl
            self._table.put_item(ptr_item)

        # Write request metadata items
        if trace_info.trace_metadata:
            rmeta_items: list[dict[str, Any]] = []
            for key, value in trace_info.trace_metadata.items():
                rmeta_item: dict[str, Any] = {
                    "PK": pk,
                    "SK": f"{SK_TRACE_PREFIX}{trace_id}#RMETA#{key}",
                    "key": key,
                    "value": value,
                }
                if ttl is not None:
                    rmeta_item["ttl"] = ttl
                rmeta_items.append(rmeta_item)
            if rmeta_items:
                self._table.batch_write(rmeta_items)

        # Write initial tag items + denormalization + FTS
        if trace_info.tags:
            for tag_key, tag_value in trace_info.tags.items():
                self._write_trace_tag(experiment_id, trace_id, tag_key, tag_value, ttl)

        # Ensure artifact location tag is set (required by MLflow trace export).
        # Matches SQLAlchemy store: always compute from experiment artifact_location.
        if MLFLOW_ARTIFACT_LOCATION not in (trace_info.tags or {}):
            exp = self.get_experiment(experiment_id)
            artifact_loc = append_to_uri_path(
                exp.artifact_location, "traces", trace_id, "artifacts"
            )
            self._write_trace_tag(
                experiment_id, trace_id, MLFLOW_ARTIFACT_LOCATION, artifact_loc, ttl
            )
            if trace_info.tags is None:
                trace_info.tags = {}
            trace_info.tags[MLFLOW_ARTIFACT_LOCATION] = artifact_loc

        # Upsert session tracker if trace has session metadata
        session_id = (trace_info.trace_metadata or {}).get("mlflow.traceSession")
        if session_id:
            self._upsert_session_tracker(
                experiment_id=experiment_id,
                session_id=session_id,
                timestamp_ms=trace_info.request_time,
                ttl=ttl,
            )

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

    @property
    def _xray_client(self) -> XRayClient:
        """Lazily initialized X-Ray client."""
        if not hasattr(self, "_xray_client_instance"):
            from mlflow_dynamodbstore.xray.client import XRayClient

            self._xray_client_instance = XRayClient(
                region=self._uri.region,
                endpoint_url=self._uri.endpoint_url,
            )
        return self._xray_client_instance

    def get_trace(
        self,
        trace_id: str,
        *,
        allow_partial: bool = False,
    ) -> Trace:
        """Fetch a trace with spans.

        Flow:
        1. Read trace info (META + tags + metadata + assessments) from DynamoDB
        2. Check for cached spans: T#<trace_id>#SPANS item
        3. If cached -> deserialize and use them
        4. If not cached -> call XRayClient.batch_get_traces([trace_id])
        5. Convert X-Ray segments -> span dicts via span_converter
        6. Cache to DynamoDB: T#<trace_id>#SPANS (JSON blob, same TTL as trace)
        7. Denormalize span attributes on META: span_types, span_statuses, span_names
        8. Write FTS items for span names
        9. Return complete Trace
        """
        import json as _json

        from mlflow.entities.trace import Trace
        from mlflow.entities.trace_data import TraceData

        from mlflow_dynamodbstore.xray.span_converter import (
            convert_xray_trace,
            span_dicts_to_mlflow_spans,
        )

        trace_info = self.get_trace_info(trace_id)
        assert trace_info.trace_location.mlflow_experiment is not None
        experiment_id = trace_info.trace_location.mlflow_experiment.experiment_id
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}"

        # Check for cached spans
        spans_sk = f"{SK_TRACE_PREFIX}{trace_id}#SPANS"
        cached = self._table.get_item(pk=pk, sk=spans_sk)

        span_dicts: list[dict[str, Any]]
        if cached is not None:
            span_dicts = _json.loads(cached["data"])
        else:
            # Fetch from X-Ray
            xray_traces = self._xray_client.batch_get_traces([trace_id])
            if xray_traces:
                span_dicts = convert_xray_trace(xray_traces[0])
            else:
                span_dicts = []

            # Get TTL from META
            meta = self._table.get_item(pk=pk, sk=sk)
            ttl = int(meta["ttl"]) if meta and "ttl" in meta else self._get_trace_ttl()

            # Cache the spans
            spans_item: dict[str, Any] = {
                "PK": pk,
                "SK": spans_sk,
                "data": _json.dumps(span_dicts),
            }
            if ttl is not None:
                spans_item["ttl"] = ttl
            self._table.put_item(spans_item)

            # Denormalize span attributes on META
            if span_dicts:
                span_types = set()
                span_statuses = set()
                span_names = set()
                for sd in span_dicts:
                    if sd.get("span_type"):
                        span_types.add(sd["span_type"])
                    if sd.get("status"):
                        span_statuses.add(sd["status"])
                    if sd.get("name"):
                        span_names.add(sd["name"])

                updates: dict[str, Any] = {}
                if span_types:
                    updates["span_types"] = span_types
                if span_statuses:
                    updates["span_statuses"] = span_statuses
                if span_names:
                    updates["span_names"] = span_names

                if updates:
                    self._table.update_item(pk=pk, sk=sk, updates=updates)

                # Write FTS items for span names
                if span_names:
                    span_names_text = " ".join(sorted(span_names))
                    fts_items = fts_items_for_text(
                        pk=pk,
                        entity_type="T",
                        entity_id=trace_id,
                        field="spans",
                        text=span_names_text,
                    )
                    if ttl is not None:
                        for item in fts_items:
                            item["ttl"] = ttl
                    self._table.batch_write(fts_items)

        # Convert span dicts to MLflow Span objects
        spans = span_dicts_to_mlflow_spans(span_dicts, trace_id)

        return Trace(info=trace_info, data=TraceData(spans=spans))

    def search_traces(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = 100,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        model_id: str | None = None,
        locations: list[str] | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
        """Search traces across experiments using parse -> plan -> execute pipeline.

        For span-level filters (field_type == "span"), uses a hybrid approach:
        1. Cached traces (those with denormalized span_types/span_names/span_statuses
           on META) are filtered via DynamoDB.
        2. Uncached traces are found via X-Ray filter expressions.
        3. Results are unioned and deduplicated.
        """
        from mlflow_dynamodbstore.dynamodb.pagination import (
            decode_page_token,
            encode_page_token,
        )
        from mlflow_dynamodbstore.dynamodb.search import (
            execute_trace_query,
            parse_trace_filter,
            plan_trace_query,
        )

        if not experiment_ids:
            experiment_ids = locations or []

        # 1. Parse filter and split into span vs non-span predicates
        predicates = parse_trace_filter(filter_string)
        span_predicates = [p for p in predicates if p.field_type == "span"]
        non_span_predicates = [p for p in predicates if p.field_type != "span"]

        # 2. Plan query using only non-span predicates
        plan = plan_trace_query(non_span_predicates, order_by)

        # 3. For each experiment: execute query
        token_data = decode_page_token(page_token)
        exp_idx = token_data.get("exp_idx", 0) if token_data else 0
        inner_token = token_data.get("inner_token") if token_data else None

        traces: list[TraceInfo] = []
        remaining = max_results
        next_page_token: str | None = None
        seen_trace_ids: set[str] = set()

        for i, exp_id in enumerate(experiment_ids[exp_idx:], start=exp_idx):
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
            current_token = inner_token if i == exp_idx else None

            # --- Phase 1: DynamoDB query (handles non-span + cached span filters) ---
            inner_next: str | None = None
            while remaining > 0:
                items, inner_next = execute_trace_query(
                    table=self._table,
                    plan=plan,
                    pk=pk,
                    max_results=remaining if not span_predicates else remaining * 3,
                    page_token=current_token,
                    predicates=non_span_predicates,
                )

                for item in items:
                    trace_id = item["trace_id"]
                    if trace_id in seen_trace_ids:
                        continue

                    # Apply span predicates on cached (denormalized) data
                    if span_predicates and not self._match_span_predicates_cached(
                        item, span_predicates
                    ):
                        continue

                    seen_trace_ids.add(trace_id)
                    self._cache.put("trace_exp", trace_id, exp_id)
                    trace_info = self._build_trace_info(exp_id, trace_id, item)
                    traces.append(trace_info)
                    remaining -= 1

                    if remaining <= 0:
                        break

                if not inner_next:
                    break
                current_token = inner_next

            # --- Phase 2: X-Ray fallback for uncached traces with span filters ---
            if span_predicates and remaining > 0:
                xray_trace_ids = self._search_xray_for_span_filters(exp_id, span_predicates)
                for xray_tid in xray_trace_ids:
                    if remaining <= 0:
                        break
                    if xray_tid in seen_trace_ids:
                        continue
                    # Verify this trace exists in DynamoDB and belongs to this experiment
                    meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{xray_tid}")
                    if meta is None:
                        continue
                    # Apply non-span post-filters on this item too
                    from mlflow_dynamodbstore.dynamodb.search import (
                        _apply_trace_post_filter,
                    )

                    if not all(
                        _apply_trace_post_filter(self._table, pk, xray_tid, meta, p)
                        for p in non_span_predicates
                    ):
                        continue

                    seen_trace_ids.add(xray_tid)
                    self._cache.put("trace_exp", xray_tid, exp_id)
                    trace_info = self._build_trace_info(exp_id, xray_tid, meta)
                    traces.append(trace_info)
                    remaining -= 1

            if remaining <= 0:
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

        return traces, next_page_token

    @staticmethod
    def _match_span_predicates_cached(
        item: dict[str, Any],
        span_predicates: list[Any],
    ) -> bool:
        """Check span predicates against denormalized span attrs on META item.

        Denormalized fields on META:
        - span_types: set of span type strings (e.g. {"LLM", "RETRIEVER"})
        - span_names: set of span name strings (e.g. {"ChatModel", "Retriever"})
        - span_statuses: set of status strings (e.g. {"OK"})

        Returns True if the item has denormalized data and all predicates match,
        or False if the item has no denormalized data (uncached).
        """
        import fnmatch as _fnmatch

        # Map span predicate keys to denormalized set fields on META
        span_key_to_field = {
            "type": "span_types",
            "name": "span_names",
            "status": "span_statuses",
        }

        has_any_denormalized = any(
            item.get(f) for f in ("span_types", "span_names", "span_statuses")
        )
        if not has_any_denormalized:
            # No cached span data -> cannot match via DynamoDB
            return False

        for pred in span_predicates:
            field_name = span_key_to_field.get(pred.key)
            if not field_name:
                # Unknown span key, skip (will need X-Ray)
                return False
            values = item.get(field_name)
            if not values:
                return False

            if pred.op == "=":
                if pred.value not in values:
                    return False
            elif pred.op == "!=":
                if pred.value in values:
                    return False
            elif pred.op in ("LIKE", "ILIKE"):
                pattern = str(pred.value).replace("%", "*").replace("_", "?")
                if pred.op == "ILIKE":
                    pattern = pattern.lower()
                matched = any(
                    _fnmatch.fnmatch(v.lower() if pred.op == "ILIKE" else v, pattern)
                    for v in values
                )
                if not matched:
                    return False
            else:
                # Unsupported op for set membership
                return False

        return True

    def _search_xray_for_span_filters(
        self,
        experiment_id: str,
        span_predicates: list[Any],
    ) -> list[str]:
        """Query X-Ray for traces matching span predicates.

        Uses the filter translator to convert span predicates to X-Ray filter
        expressions, then calls get_trace_summaries.

        Returns a list of trace IDs found via X-Ray.
        """
        import datetime

        from mlflow_dynamodbstore.xray.annotation_config import DEFAULT_ANNOTATION_CONFIG
        from mlflow_dynamodbstore.xray.filter_translator import translate_span_filters

        # Map span predicate keys to annotation config keys
        span_key_to_annotation = {
            "type": "mlflow.spanType",
            "name": "name",
            "status": "status",
        }

        # Remap span predicates to use annotation config keys
        from mlflow_dynamodbstore.dynamodb.search import FilterPredicate

        remapped = []
        for pred in span_predicates:
            ann_key = span_key_to_annotation.get(pred.key, pred.key)
            remapped.append(
                FilterPredicate(
                    field_type=pred.field_type,
                    key=ann_key,
                    op=pred.op,
                    value=pred.value,
                )
            )

        xray_filter, _remaining = translate_span_filters(remapped, DEFAULT_ANNOTATION_CONFIG)
        if not xray_filter:
            return []

        # Use a reasonable time window (last 30 days)
        end_time = datetime.datetime.now(tz=datetime.UTC)
        start_time = end_time - datetime.timedelta(days=30)

        try:
            summaries = self._xray_client.get_trace_summaries(
                start_time=start_time,
                end_time=end_time,
                filter_expression=xray_filter,
            )
        except Exception:
            return []

        return [s["Id"] for s in summaries if "Id" in s]

    def _build_trace_info(
        self,
        experiment_id: str,
        trace_id: str,
        meta: dict[str, Any],
    ) -> TraceInfo:
        """Build a TraceInfo from a META item + sub-item lookups."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

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
        ttl: int | None,
    ) -> None:
        """Write a trace tag item with TTL, optionally denormalize, and write FTS items."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        tag_sk = f"{SK_TRACE_PREFIX}{trace_id}#TAG#{tag_key}"

        # Read existing tag to support FTS diff on overwrite
        old_tag = self._table.get_item(pk=pk, sk=tag_sk)
        old_value: str | None = old_tag["value"] if old_tag else None

        item: dict[str, Any] = {
            "PK": pk,
            "SK": tag_sk,
            "key": tag_key,
            "value": tag_value,
        }
        if ttl is not None:
            item["ttl"] = ttl
        self._table.put_item(item)
        if self._config.should_denormalize(experiment_id, tag_key):
            self._denormalize_tag(pk, f"{SK_TRACE_PREFIX}{trace_id}", tag_key, tag_value)
        # Update FTS items for tag value if configured
        if self._config.should_trigram("trace_tag_value") and (tag_value or old_value):
            field_suffix = f"#{tag_key}"
            entity_prefix = f"T#{trace_id}{field_suffix}"
            levels = ("W", "3")
            tokens_to_add, tokens_to_remove = fts_diff(old_value, tag_value, levels)

            # Delete removed FTS items
            if tokens_to_remove:
                rev_prefix = f"{SK_FTS_REV_PREFIX}{entity_prefix}#"
                rev_items = self._table.query(pk=pk, sk_prefix=rev_prefix)
                for rev_item in rev_items:
                    rev_sk = rev_item["SK"]
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
                new_fts_items: list[dict[str, Any]] = []
                for lvl, tok in tokens_to_add:
                    forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#{entity_prefix}"
                    reverse_sk = f"{SK_FTS_REV_PREFIX}{entity_prefix}#{lvl}#{tok}"
                    new_fwd: dict[str, Any] = {"PK": pk, "SK": forward_sk}
                    new_rev: dict[str, Any] = {"PK": pk, "SK": reverse_sk}
                    if ttl is not None:
                        new_fwd["ttl"] = ttl
                        new_rev["ttl"] = ttl
                    new_fts_items.append(new_fwd)
                    new_fts_items.append(new_rev)
                self._table.batch_write(new_fts_items)

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
        ttl = int(meta["ttl"]) if "ttl" in meta else self._get_trace_ttl()
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
            }
            if ttl is not None:
                rmeta_item["ttl"] = ttl
            self._table.put_item(rmeta_item)

    def unlink_traces_from_run(self, trace_ids: list[str], run_id: str) -> None:
        """Unlink traces from a run by deleting mlflow.sourceRun RMETA items."""
        for trace_id in trace_ids:
            try:
                experiment_id = self._resolve_trace_experiment(trace_id)
            except MlflowException:
                continue
            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            sk = f"{SK_TRACE_PREFIX}{trace_id}#RMETA#{TraceMetadataKey.SOURCE_RUN}"
            self._table.delete_item(pk=pk, sk=sk)

    def link_prompts_to_trace(self, trace_id: str, prompt_versions: list[PromptVersion]) -> None:
        """Link prompt versions to a trace by writing mlflow.promptVersions tag."""
        import json as _json

        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
        if meta is None:
            raise MlflowException(
                f"Trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        ttl = int(meta["ttl"]) if "ttl" in meta else self._get_trace_ttl()

        versions_json = _json.dumps(
            [{"name": pv.name, "version": pv.version} for pv in prompt_versions]
        )
        self._write_trace_tag(experiment_id, trace_id, "mlflow.promptVersions", versions_json, ttl)

    def batch_get_trace_infos(
        self, trace_ids: list[str], location: str | None = None
    ) -> list[TraceInfo]:
        """Get trace metadata for given trace IDs without loading spans."""
        if not trace_ids:
            return []

        seen: set[str] = set()
        results: list[TraceInfo] = []

        for trace_id in trace_ids:
            if trace_id in seen:
                continue
            seen.add(trace_id)

            try:
                if location:
                    experiment_id = location
                else:
                    experiment_id = self._resolve_trace_experiment(trace_id)
            except MlflowException:
                continue  # Skip non-existent traces

            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
            if meta is None:
                continue

            trace_info = self._build_trace_info(experiment_id, trace_id, meta)
            results.append(trace_info)

        return results

    def batch_get_traces(self, trace_ids: list[str], location: str | None = None) -> list[Trace]:
        """Get complete traces with spans for given trace IDs."""
        import json as _json

        from mlflow.entities.trace import Trace
        from mlflow.entities.trace_data import TraceData

        from mlflow_dynamodbstore.xray.span_converter import span_dicts_to_mlflow_spans

        if not trace_ids:
            return []

        seen: set[str] = set()
        results: list[Trace] = []

        for trace_id in trace_ids:
            if trace_id in seen:
                continue
            seen.add(trace_id)

            try:
                if location:
                    experiment_id = location
                else:
                    experiment_id = self._resolve_trace_experiment(trace_id)
            except MlflowException:
                continue

            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
            if meta is None:
                continue

            trace_info = self._build_trace_info(experiment_id, trace_id, meta)

            # Read cached spans
            spans_sk = f"{SK_TRACE_PREFIX}{trace_id}#SPANS"
            cached = self._table.get_item(pk=pk, sk=spans_sk)
            if cached is not None:
                span_dicts = _json.loads(cached["data"])
                # Handle both V3 format (Span.to_dict) and X-Ray format
                if span_dicts and "start_time_unix_nano" in span_dicts[0]:
                    # V3 format: use Span.from_dict
                    from mlflow.entities.span import Span as SpanEntity

                    spans = [SpanEntity.from_dict(sd) for sd in span_dicts]
                else:
                    # X-Ray converter format
                    spans = span_dicts_to_mlflow_spans(span_dicts, trace_id)
            else:
                spans = []

            results.append(Trace(info=trace_info, data=TraceData(spans=spans)))

        return results

    # ------------------------------------------------------------------
    # Assessment CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _assessment_fts_text(assessment: Assessment) -> str | None:
        """Extract searchable text from an assessment's value (feedback or expectation)."""
        if assessment.feedback and assessment.feedback.value is not None:
            val = assessment.feedback.value
            return str(val) if not isinstance(val, str) else val
        if assessment.expectation and assessment.expectation.value is not None:
            val = assessment.expectation.value
            return str(val) if not isinstance(val, str) else val
        return None

    def _write_assessment_fts(
        self,
        pk: str,
        trace_id: str,
        assessment_id: str,
        text: str,
        ttl: int | None,
    ) -> None:
        """Write FTS forward + reverse items for an assessment's value text."""
        field = f"assess_{assessment_id}"
        fts_items = fts_items_for_text(
            pk=pk,
            entity_type="T",
            entity_id=trace_id,
            field=field,
            text=text,
        )
        if ttl is not None:
            for item in fts_items:
                item["ttl"] = ttl
        if fts_items:
            self._table.batch_write(fts_items)

    def create_assessment(self, assessment: Assessment) -> Assessment:
        """Create a new assessment for a trace."""
        trace_id = assessment.trace_id
        if trace_id is None:
            raise MlflowException(
                "Assessment must have a trace_id.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        # Read TTL from the trace META item
        meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
        if meta is None:
            raise MlflowException(
                f"Trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        ttl = int(meta["ttl"]) if "ttl" in meta else self._get_trace_ttl()

        # Generate assessment ID
        assessment_id = generate_ulid()
        now_ms = int(time.time() * 1000)

        # Build the assessment item storing the full serialized assessment dict
        assess_dict = assessment.to_dictionary()
        assess_dict["assessment_id"] = assessment_id
        assess_dict["create_time"] = assess_dict.get(
            "create_time"
        ) or milliseconds_to_proto_timestamp(now_ms)
        assess_dict["last_update_time"] = assess_dict.get(
            "last_update_time"
        ) or milliseconds_to_proto_timestamp(now_ms)

        sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{assessment_id}"
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "data": assess_dict,
        }
        if ttl is not None:
            item["ttl"] = ttl
        self._table.put_item(item)

        # Write FTS items for the assessment value text
        fts_text = self._assessment_fts_text(assessment)
        if fts_text:
            self._write_assessment_fts(pk, trace_id, assessment_id, fts_text, ttl)

        # Return the assessment with the generated ID
        result = Assessment.from_dictionary(assess_dict)
        return result

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """Fetch an assessment by trace ID and assessment ID."""
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{assessment_id}"

        item = self._table.get_item(pk=pk, sk=sk)
        if item is None:
            raise MlflowException(
                f"Assessment '{assessment_id}' for trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return Assessment.from_dictionary(item["data"])

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: str | None = None,
        expectation: str | None = None,
        feedback: str | None = None,
        rationale: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Assessment:
        """Update mutable fields of an assessment."""
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{assessment_id}"

        item = self._table.get_item(pk=pk, sk=sk)
        if item is None:
            raise MlflowException(
                f"Assessment '{assessment_id}' for trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        ttl = int(item["ttl"]) if "ttl" in item else self._get_trace_ttl()

        assess_dict = item["data"]

        # Capture old text for FTS diff
        old_assessment = Assessment.from_dictionary(assess_dict)
        old_fts_text = self._assessment_fts_text(old_assessment)

        # Apply updates
        now_ms = int(time.time() * 1000)
        assess_dict["last_update_time"] = milliseconds_to_proto_timestamp(now_ms)

        if name is not None:
            assess_dict["assessment_name"] = name
        if feedback is not None:
            assess_dict["feedback"] = {"value": feedback}
            assess_dict.pop("expectation", None)
        if expectation is not None:
            assess_dict["expectation"] = {"value": expectation}
            assess_dict.pop("feedback", None)
        if rationale is not None:
            assess_dict["rationale"] = rationale
        if metadata is not None:
            assess_dict["metadata"] = metadata

        # Write updated item
        item["data"] = assess_dict
        self._table.put_item(item)

        # FTS diff
        updated_assessment = Assessment.from_dictionary(assess_dict)
        new_fts_text = self._assessment_fts_text(updated_assessment)
        field = f"assess_{assessment_id}"

        if old_fts_text or new_fts_text:
            levels = ("W", "3")
            tokens_to_add, tokens_to_remove = fts_diff(old_fts_text, new_fts_text or "", levels)
            field_suffix = f"#{field}"
            entity_prefix = f"T#{trace_id}{field_suffix}"

            # Delete removed FTS items
            if tokens_to_remove:
                rev_prefix = f"{SK_FTS_REV_PREFIX}{entity_prefix}#"
                rev_items = self._table.query(pk=pk, sk_prefix=rev_prefix)
                for rev_item in rev_items:
                    rev_sk = rev_item["SK"]
                    suffix = rev_sk[len(SK_FTS_REV_PREFIX) + len(entity_prefix) + 1 :]
                    parts = suffix.split("#", 1)
                    if len(parts) == 2:
                        lvl, tok = parts[0], parts[1]
                        if (lvl, tok) in tokens_to_remove:
                            forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#{entity_prefix}"
                            self._table.delete_item(pk=pk, sk=forward_sk)
                            self._table.delete_item(pk=pk, sk=rev_sk)

            # Write new FTS items
            if tokens_to_add:
                new_fts_items: list[dict[str, Any]] = []
                for lvl, tok in tokens_to_add:
                    forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#{entity_prefix}"
                    reverse_sk = f"{SK_FTS_REV_PREFIX}{entity_prefix}#{lvl}#{tok}"
                    new_fwd: dict[str, Any] = {"PK": pk, "SK": forward_sk}
                    new_rev: dict[str, Any] = {"PK": pk, "SK": reverse_sk}
                    if ttl is not None:
                        new_fwd["ttl"] = ttl
                        new_rev["ttl"] = ttl
                    new_fts_items.append(new_fwd)
                    new_fts_items.append(new_rev)
                self._table.batch_write(new_fts_items)

        return updated_assessment

    def delete_assessment(self, trace_id: str, assessment_id: str) -> None:
        """Delete an assessment and clean up its FTS items."""
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{assessment_id}"

        item = self._table.get_item(pk=pk, sk=sk)
        if item is None:
            raise MlflowException(
                f"Assessment '{assessment_id}' for trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Delete assessment item
        self._table.delete_item(pk=pk, sk=sk)

        # Clean up FTS items via reverse index
        field = f"assess_{assessment_id}"
        self._delete_fts_for_entity_field(pk=pk, entity_type="T", entity_id=trace_id, field=field)

    # ------------------------------------------------------------------
    # Scorer CRUD
    # ------------------------------------------------------------------

    def _resolve_scorer_id(self, experiment_id: str, name: str) -> str | None:
        """Resolve scorer name to scorer_id via GSI3. Returns None if not found."""
        results = self._table.query(
            pk=f"{GSI3_SCOR_NAME_PREFIX}{self._workspace}#{experiment_id}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if not results:
            return None
        return str(results[0][GSI3_SK])  # gsi3sk holds the scorer_id directly

    def register_scorer(
        self, experiment_id: str, name: str, serialized_scorer: str
    ) -> ScorerVersion:
        self.get_experiment(experiment_id)  # verify exists
        now_ms = int(time.time() * 1000)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        existing_scorer_id = self._resolve_scorer_id(experiment_id, name)

        if existing_scorer_id is None:
            scorer_id = generate_ulid()
            meta_sk = f"{SK_SCORER_PREFIX}{scorer_id}"
            meta_item: dict[str, Any] = {
                "PK": pk,
                "SK": meta_sk,
                "scorer_name": name,
                "scorer_id": scorer_id,
                "latest_version": 1,
                "workspace": self._workspace,
                GSI1_PK: f"{GSI1_SCOR_PREFIX}{scorer_id}",
                GSI1_SK: pk,
                GSI3_PK: f"{GSI3_SCOR_NAME_PREFIX}{self._workspace}#{experiment_id}#{name}",
                GSI3_SK: scorer_id,
                LSI3_SK: name.lower(),
            }
            try:
                self._table.put_item(meta_item, condition="attribute_not_exists(SK)")
            except Exception:
                # Race: another registration won. Retry via existing path.
                existing_scorer_id = self._resolve_scorer_id(experiment_id, name)
                if existing_scorer_id is None:
                    raise

            if existing_scorer_id is None:
                # New scorer path succeeded
                version = 1
                padded = f"{version:010d}"
                ver_item: dict[str, Any] = {
                    "PK": pk,
                    "SK": f"{meta_sk}#V#{padded}",
                    "scorer_version": version,
                    "serialized_scorer": serialized_scorer,
                    "creation_time": now_ms,
                }
                self._table.put_item(ver_item)
                return _ScorerVersionCompat(
                    experiment_id=experiment_id,
                    scorer_name=name,
                    scorer_version=version,
                    serialized_scorer=serialized_scorer,
                    creation_time=now_ms,
                    scorer_id=scorer_id,
                )

        # Existing scorer path (including race retry)
        scorer_id = existing_scorer_id
        meta_sk = f"{SK_SCORER_PREFIX}{scorer_id}"
        updated = self._table.add_attribute(pk=pk, sk=meta_sk, attribute="latest_version", value=1)
        version = int(updated["latest_version"])
        padded = f"{version:010d}"
        ver_item = {
            "PK": pk,
            "SK": f"{meta_sk}#V#{padded}",
            "scorer_version": version,
            "serialized_scorer": serialized_scorer,
            "creation_time": now_ms,
        }
        self._table.put_item(ver_item)
        return _ScorerVersionCompat(
            experiment_id=experiment_id,
            scorer_name=name,
            scorer_version=version,
            serialized_scorer=serialized_scorer,
            creation_time=now_ms,
            scorer_id=scorer_id,
        )

    def get_scorer(
        self, experiment_id: str, name: str, version: int | None = None
    ) -> ScorerVersion:
        scorer_id = self._resolve_scorer_id(experiment_id, name)
        if scorer_id is None:
            raise MlflowException(
                f"Scorer '{name}' not found in experiment '{experiment_id}'.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        meta_sk = f"{SK_SCORER_PREFIX}{scorer_id}"

        if version is not None:
            padded = f"{version:010d}"
            item = self._table.get_item(pk=pk, sk=f"{meta_sk}#V#{padded}")
            if item is None:
                raise MlflowException(
                    f"Scorer '{name}' version {version} not found.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
        else:
            # AP3: latest version by SK sort descending
            items = self._table.query(pk=pk, sk_prefix=f"{meta_sk}#V#", scan_forward=False, limit=1)
            if not items:
                raise MlflowException(
                    f"Scorer '{name}' has no versions.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            item = items[0]

        # Read META for scorer_name (in case name casing differs)
        meta = self._table.get_item(pk=pk, sk=meta_sk)
        return _ScorerVersionCompat(
            experiment_id=experiment_id,
            scorer_name=meta["scorer_name"] if meta else name,
            scorer_version=int(item["scorer_version"]),
            serialized_scorer=item["serialized_scorer"],
            creation_time=int(item["creation_time"]),
            scorer_id=scorer_id,
        )

    def list_scorers(self, experiment_id: str) -> list[ScorerVersion]:
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        # Query all SCOR# items and filter to META items in Python.
        # META items have SK = "SCOR#<ulid>" (no #V# or #OSCFG suffix).
        items = self._table.query(pk=pk, sk_prefix=SK_SCORER_PREFIX)
        meta_items = [item for item in items if "#" not in item["SK"][len(SK_SCORER_PREFIX) :]]

        result: list[ScorerVersion] = []
        for meta in meta_items:
            scorer_id = meta["SK"][len(SK_SCORER_PREFIX) :]
            # AP3: get latest version
            versions = self._table.query(
                pk=pk,
                sk_prefix=f"{SK_SCORER_PREFIX}{scorer_id}#V#",
                scan_forward=False,
                limit=1,
            )
            if versions:
                ver = versions[0]
                result.append(
                    _ScorerVersionCompat(
                        experiment_id=experiment_id,
                        scorer_name=meta["scorer_name"],
                        scorer_version=int(ver["scorer_version"]),
                        serialized_scorer=ver["serialized_scorer"],
                        creation_time=int(ver["creation_time"]),
                        scorer_id=scorer_id,
                    )
                )
        return result

    def list_scorer_versions(self, experiment_id: str, name: str) -> list[ScorerVersion]:
        scorer_id = self._resolve_scorer_id(experiment_id, name)
        if scorer_id is None:
            raise MlflowException(
                f"Scorer '{name}' not found in experiment '{experiment_id}'.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        # AP5: all versions ascending
        items = self._table.query(
            pk=pk,
            sk_prefix=f"{SK_SCORER_PREFIX}{scorer_id}#V#",
            scan_forward=True,
        )
        # Read META for canonical scorer_name
        meta = self._table.get_item(pk=pk, sk=f"{SK_SCORER_PREFIX}{scorer_id}")
        scorer_name = meta["scorer_name"] if meta else name
        return [
            _ScorerVersionCompat(
                experiment_id=experiment_id,
                scorer_name=scorer_name,
                scorer_version=int(item["scorer_version"]),
                serialized_scorer=item["serialized_scorer"],
                creation_time=int(item["creation_time"]),
                scorer_id=scorer_id,
            )
            for item in items
        ]

    def delete_scorer(self, experiment_id: str, name: str, version: int | None = None) -> None:
        scorer_id = self._resolve_scorer_id(experiment_id, name)
        if scorer_id is None:
            raise MlflowException(
                f"Scorer '{name}' not found in experiment '{experiment_id}'.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        meta_sk = f"{SK_SCORER_PREFIX}{scorer_id}"

        if version is None:
            # AP6: delete all items (META + versions + config)
            items = self._table.query(pk=pk, sk_prefix=f"{SK_SCORER_PREFIX}{scorer_id}")
            if items:
                self._table.batch_delete([{"PK": pk, "SK": item["SK"]} for item in items])
        else:
            # AP7: delete single version
            padded = f"{version:010d}"
            ver_sk = f"{meta_sk}#V#{padded}"
            item = self._table.get_item(pk=pk, sk=ver_sk)
            if item is None:
                raise MlflowException(
                    f"Scorer '{name}' version {version} not found.",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            self._table.delete_item(pk=pk, sk=ver_sk)

            # Check if any versions remain
            remaining = self._table.query(pk=pk, sk_prefix=f"{meta_sk}#V#", limit=1)
            if not remaining:
                # Delete META and config too
                self._table.delete_item(pk=pk, sk=meta_sk)
                self._table.delete_item(pk=pk, sk=f"{meta_sk}{SK_SCORER_OSCFG_SUFFIX}")
            else:
                # Update latest_version cache if needed
                latest = self._table.query(
                    pk=pk,
                    sk_prefix=f"{meta_sk}#V#",
                    scan_forward=False,
                    limit=1,
                )
                if latest:
                    new_max = int(latest[0]["scorer_version"])
                    self._table.update_item(
                        pk=pk,
                        sk=meta_sk,
                        updates={"latest_version": new_max},
                    )

    def upsert_online_scoring_config(
        self,
        experiment_id: str,
        scorer_name: str,
        sample_rate: float,
        filter_string: str | None = None,
    ) -> OnlineScoringConfig:
        from mlflow.genai.scorers.online.entities import OnlineScoringConfig

        if not (0.0 <= sample_rate <= 1.0):
            raise MlflowException(
                f"sample_rate must be in [0.0, 1.0], got {sample_rate}.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        scorer_id = self._resolve_scorer_id(experiment_id, scorer_name)
        if scorer_id is None:
            raise MlflowException(
                f"Scorer '{scorer_name}' not found in experiment '{experiment_id}'.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        config_id = generate_ulid()
        config_sk = f"{SK_SCORER_PREFIX}{scorer_id}{SK_SCORER_OSCFG_SUFFIX}"

        item: dict[str, Any] = {
            "PK": pk,
            "SK": config_sk,
            "online_scoring_config_id": config_id,
            "scorer_id": scorer_id,
            "sample_rate": Decimal(str(sample_rate)),
            "experiment_id": experiment_id,
        }
        if filter_string is not None:
            item["filter_string"] = filter_string

        # GSI2: only index when active (sample_rate > 0)
        if sample_rate > 0:
            item[GSI2_PK] = f"{GSI2_ACTIVE_SCORERS_PREFIX}{self._workspace}"
            item[GSI2_SK] = scorer_id

        self._table.put_item(item)  # atomic overwrite (fixed SK)

        return OnlineScoringConfig(
            online_scoring_config_id=config_id,
            scorer_id=scorer_id,
            sample_rate=sample_rate,
            experiment_id=experiment_id,
            filter_string=filter_string,
        )

    def get_online_scoring_configs(self, scorer_ids: list[str]) -> list[OnlineScoringConfig]:
        from mlflow.genai.scorers.online.entities import OnlineScoringConfig

        configs: list[OnlineScoringConfig] = []
        for scorer_id in scorer_ids:
            # AP11: resolve scorer_id → experiment_id via GSI1
            results = self._table.query(
                pk=f"{GSI1_SCOR_PREFIX}{scorer_id}",
                index_name="gsi1",
                limit=1,
            )
            if not results:
                continue
            exp_id = results[0][GSI1_SK][len(PK_EXPERIMENT_PREFIX) :]
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
            config_sk = f"{SK_SCORER_PREFIX}{scorer_id}{SK_SCORER_OSCFG_SUFFIX}"
            item = self._table.get_item(pk=pk, sk=config_sk)
            if item is None:
                continue
            configs.append(
                OnlineScoringConfig(
                    online_scoring_config_id=item["online_scoring_config_id"],
                    scorer_id=item["scorer_id"],
                    sample_rate=float(item["sample_rate"]),
                    experiment_id=item["experiment_id"],
                    filter_string=item.get("filter_string"),
                )
            )
        return configs

    def get_active_online_scorers(self) -> list[OnlineScorer]:
        from mlflow.genai.scorers.online.entities import (
            OnlineScorer,
            OnlineScoringConfig,
        )

        # AP9: query GSI2 for active configs
        items = self._table.query(
            pk=f"{GSI2_ACTIVE_SCORERS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )

        # Deduplicate by scorer_id
        seen: set[str] = set()
        unique_items: list[dict[str, Any]] = []
        for item in items:
            sid = item["scorer_id"]
            if sid not in seen:
                seen.add(sid)
                unique_items.append(item)

        result: list[OnlineScorer] = []
        for item in unique_items:
            scorer_id = item["scorer_id"]
            exp_id = item["experiment_id"]
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"

            # Get META for scorer_name
            meta = self._table.get_item(pk=pk, sk=f"{SK_SCORER_PREFIX}{scorer_id}")
            if meta is None:
                continue

            # AP3: get latest version for serialized_scorer
            versions = self._table.query(
                pk=pk,
                sk_prefix=f"{SK_SCORER_PREFIX}{scorer_id}#V#",
                scan_forward=False,
                limit=1,
            )
            if not versions:
                continue

            config = OnlineScoringConfig(
                online_scoring_config_id=item["online_scoring_config_id"],
                scorer_id=scorer_id,
                sample_rate=float(item["sample_rate"]),
                experiment_id=exp_id,
                filter_string=item.get("filter_string"),
            )
            result.append(
                OnlineScorer(
                    name=meta["scorer_name"],
                    serialized_scorer=versions[0]["serialized_scorer"],
                    online_config=config,
                )
            )
        return result

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        """Delete traces and all their sub-items (tags, metadata, assessments, FTS)."""
        if not trace_ids:
            return 0

        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        deleted = 0

        for trace_id in trace_ids:
            # 1. Query all trace sub-items: SK begins_with T#<trace_id>
            trace_prefix = f"{SK_TRACE_PREFIX}{trace_id}"
            trace_items = self._table.query(pk=pk, sk_prefix=trace_prefix)

            # 2. Query FTS_REV items for this trace to find forward FTS items
            fts_rev_prefix = f"{SK_FTS_REV_PREFIX}T#{trace_id}"
            fts_rev_items = self._table.query(pk=pk, sk_prefix=fts_rev_prefix)

            # 3. Derive forward FTS SKs from reverse items and delete them
            for rev_item in fts_rev_items:
                rev_sk = rev_item["SK"]
                # Everything after "FTS_REV#"
                after_rev_prefix = rev_sk[len(SK_FTS_REV_PREFIX) :]
                # after_rev_prefix: T#<trace_id>[#<field>]#<level>#<token>
                # We need to split into entity_prefix and level#token.
                # entity_prefix always starts with T#<trace_id>.
                # After T#<trace_id>, there may be #<field> or directly #<level>#<token>.
                # Level is always a single character (W or 3).
                # So we look for the pattern where after the entity part,
                # we have #<single_char>#<rest> where single_char is the level.
                base = f"T#{trace_id}"
                rest = after_rev_prefix[len(base) :]
                # rest starts with '#'
                # Split: could be #<field>#<level>#<token> or #<level>#<token>
                # Level chars: W, 3
                parts = rest.lstrip("#").split("#")
                # Try to find the level marker
                # If parts[0] is a known level (W or 3), then no field
                # Otherwise parts[0] is field, parts[1] is level
                if parts[0] in ("W", "3"):
                    entity_prefix = base
                    lvl = parts[0]
                    tok = "#".join(parts[1:])
                else:
                    entity_prefix = f"{base}#{parts[0]}"
                    lvl = parts[1]
                    tok = "#".join(parts[2:])

                forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#{entity_prefix}"
                self._table.delete_item(pk=pk, sk=forward_sk)
                self._table.delete_item(pk=pk, sk=rev_sk)

            # 4. Delete all trace sub-items (META, tags, RMETA, assessments, CLIENTPTR)
            for item in trace_items:
                self._table.delete_item(pk=pk, sk=item["SK"])

            deleted += 1

        return deleted

    # ------------------------------------------------------------------
    # Evaluation Dataset CRUD
    # ------------------------------------------------------------------

    def _compute_dataset_digest(self, name: str, last_update_time: int) -> str:
        """Compute a short digest for a dataset from its name and last_update_time."""
        return hashlib.sha256(f"{name}:{last_update_time}".encode()).hexdigest()[:8]

    def create_dataset(
        self,
        name: str,
        tags: dict[str, str] | None = None,
        experiment_ids: list[str] | None = None,
    ) -> EvaluationDataset:
        """Create a new evaluation dataset and return it."""
        # Check name uniqueness via GSI3
        existing = self._table.query(
            pk=f"{GSI3_DS_NAME_PREFIX}{self._workspace}#{name.lower()}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Dataset with name '{name}' already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        now_ms = int(time.time() * 1000)
        dataset_id = f"eval_{generate_ulid()}"
        digest = self._compute_dataset_digest(name, now_ms)
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"

        # Write META item
        meta_item: dict[str, Any] = {
            "PK": pk,
            "SK": SK_DATASET_META,
            "name": name,
            "digest": digest,
            "created_time": now_ms,
            "last_update_time": now_ms,
            "workspace": self._workspace,
            "tags": tags or {},
            # LSI projections
            LSI1_SK: f"{now_ms:020d}",
            LSI2_SK: now_ms,
            LSI3_SK: name.lower(),
            # GSI2: list all datasets in workspace
            GSI2_PK: f"{GSI2_DS_LIST_PREFIX}{self._workspace}",
            GSI2_SK: dataset_id,
            # GSI3: name uniqueness
            GSI3_PK: f"{GSI3_DS_NAME_PREFIX}{self._workspace}#{name.lower()}",
            GSI3_SK: dataset_id,
        }
        self._table.put_item(meta_item)

        # Write tag items
        if tags:
            for key, value in tags.items():
                tag_item: dict[str, Any] = {
                    "PK": pk,
                    "SK": f"{SK_DATASET_TAG_PREFIX}{key}",
                    "key": key,
                    "value": value,
                }
                self._table.put_item(tag_item)

        # Write experiment link items
        if experiment_ids:
            for exp_id in experiment_ids:
                exp_link_item: dict[str, Any] = {
                    "PK": pk,
                    "SK": f"{SK_DATASET_EXP_PREFIX}{exp_id}",
                    # GSI1 reverse lookup: from experiment -> datasets
                    GSI1_PK: f"{GSI1_DS_EXP_PREFIX}{exp_id}",
                    GSI1_SK: dataset_id,
                }
                self._table.put_item(exp_link_item)

        return EvaluationDataset(
            dataset_id=dataset_id,
            name=name,
            digest=digest,
            created_time=now_ms,
            last_update_time=now_ms,
            tags=tags or {},
        )

    def get_dataset(self, dataset_id: str) -> EvaluationDataset:
        """Fetch an evaluation dataset by ID."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is None:
            raise MlflowException(
                f"Dataset '{dataset_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Load tags from tag items
        tag_items = self._table.query(pk=pk, sk_prefix=SK_DATASET_TAG_PREFIX)
        tags = {item["key"]: item["value"] for item in tag_items}

        # Load experiment IDs
        experiment_ids = self.get_dataset_experiment_ids(dataset_id)

        ds = EvaluationDataset(
            dataset_id=dataset_id,
            name=meta["name"],
            digest=meta["digest"],
            created_time=int(meta["created_time"]),
            last_update_time=int(meta["last_update_time"]),
            tags=tags,
            profile=meta.get("profile"),
        )
        ds.experiment_ids = experiment_ids
        return ds

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete an evaluation dataset and all its sub-items."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        all_items = self._table.query(pk=pk)
        if not all_items:
            # Idempotent: nothing to delete
            return
        keys = [{"PK": item["PK"], "SK": item["SK"]} for item in all_items]
        self._table.batch_delete(keys)

    def get_dataset_experiment_ids(self, dataset_id: str) -> list[str]:
        """Return experiment IDs associated with a dataset."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        items = self._table.query(pk=pk, sk_prefix=SK_DATASET_EXP_PREFIX)
        return [item["SK"][len(SK_DATASET_EXP_PREFIX) :] for item in items]

    def search_datasets(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = 1000,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[EvaluationDataset]:
        """Search evaluation datasets with optional filtering, ordering, and pagination.

        Args:
            experiment_ids: Limit results to datasets linked to these experiments.
            filter_string: Filter expression, supports ``name LIKE 'pattern'``.
            max_results: Maximum number of results per page.
            order_by: Ordering criteria, e.g. ``["name ASC"]``.
            page_token: Opaque token for fetching the next page.

        Returns:
            A PagedList of EvaluationDataset objects.
        """
        import re

        from mlflow_dynamodbstore.dynamodb.pagination import (
            decode_page_token,
            encode_page_token,
        )

        # Decode pagination state
        token_state = decode_page_token(page_token)
        offset = token_state["offset"] if token_state else 0

        # --- Collect all matching dataset META items ---
        if experiment_ids:
            # AP5 + AP13: query GSI1 for each experiment, then get META items
            dataset_ids: list[str] = []
            seen: set[str] = set()
            for exp_id in experiment_ids:
                gsi1_pk = f"{GSI1_DS_EXP_PREFIX}{exp_id}"
                link_items = self._table.query(pk=gsi1_pk, index_name="gsi1")
                for item in link_items:
                    did = item[GSI1_SK]
                    if did not in seen:
                        seen.add(did)
                        dataset_ids.append(did)

            all_datasets: list[EvaluationDataset] = []
            for did in dataset_ids:
                pk = f"{PK_DATASET_PREFIX}{did}"
                meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
                if meta is None:
                    continue
                all_datasets.append(
                    EvaluationDataset(
                        dataset_id=did,
                        name=meta["name"],
                        digest=meta["digest"],
                        created_time=int(meta["created_time"]),
                        last_update_time=int(meta["last_update_time"]),
                        tags=meta.get("tags") or {},
                    )
                )
        else:
            # AP4: query GSI2 for all datasets in workspace
            gsi2_pk = f"{GSI2_DS_LIST_PREFIX}{self._workspace}"
            meta_items = self._table.query(pk=gsi2_pk, index_name="gsi2")
            all_datasets = []
            for item in meta_items:
                did = item[GSI2_SK]
                pk = f"{PK_DATASET_PREFIX}{did}"
                meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
                if meta is None:
                    continue
                all_datasets.append(
                    EvaluationDataset(
                        dataset_id=did,
                        name=meta["name"],
                        digest=meta["digest"],
                        created_time=int(meta["created_time"]),
                        last_update_time=int(meta["last_update_time"]),
                        tags=meta.get("tags") or {},
                    )
                )

        # --- Apply filter_string in-memory ---
        if filter_string:
            # Support: name LIKE 'pattern%' or name LIKE '%pattern%'
            like_match = re.match(
                r"""name\s+LIKE\s+['"](.+)['"]\s*$""", filter_string.strip(), re.IGNORECASE
            )
            if like_match:
                pattern = like_match.group(1)
                if pattern.startswith("%") and pattern.endswith("%"):
                    substring = pattern.strip("%")
                    all_datasets = [d for d in all_datasets if substring in d.name]
                elif pattern.endswith("%"):
                    prefix = pattern.rstrip("%")
                    all_datasets = [d for d in all_datasets if d.name.startswith(prefix)]
                elif pattern.startswith("%"):
                    suffix = pattern.lstrip("%")
                    all_datasets = [d for d in all_datasets if d.name.endswith(suffix)]
                else:
                    all_datasets = [d for d in all_datasets if d.name == pattern]

        # --- Apply order_by in-memory ---
        if order_by:
            for criterion in reversed(order_by):
                parts = criterion.strip().split()
                field = parts[0].lower()
                direction = parts[1].upper() if len(parts) > 1 else "ASC"
                reverse = direction == "DESC"
                if field == "name":
                    all_datasets.sort(key=lambda d: d.name, reverse=reverse)
                elif field == "created_time":
                    all_datasets.sort(key=lambda d: d.created_time, reverse=reverse)
                elif field == "last_update_time":
                    all_datasets.sort(key=lambda d: d.last_update_time, reverse=reverse)

        # --- Apply pagination ---
        total = len(all_datasets)
        page = all_datasets[offset : offset + max_results]
        next_offset = offset + len(page)
        next_token = encode_page_token({"offset": next_offset}) if next_offset < total else None

        return PagedList(page, next_token)

    def set_dataset_tags(self, dataset_id: str, tags: dict[str, str]) -> None:
        """Set (upsert) one or more tags on an evaluation dataset."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        for key, value in tags.items():
            # Write individual tag item (overwrite = upsert)
            self._table.put_item(
                {
                    "PK": pk,
                    "SK": f"{SK_DATASET_TAG_PREFIX}{key}",
                    "key": key,
                    "value": value,
                }
            )
            # Update denormalized tags map on META item
            self._denormalize_tag(pk=pk, sk=SK_DATASET_META, tag_key=key, tag_value=value)

        # Update last_update_time and digest
        now_ms = int(time.time() * 1000)
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is not None:
            digest = self._compute_dataset_digest(meta["name"], now_ms)
            self._table.update_item(
                pk=pk,
                sk=SK_DATASET_META,
                updates={"last_update_time": now_ms, "digest": digest},
            )

    def delete_dataset_tag(self, dataset_id: str, key: str) -> None:
        """Delete a single tag from an evaluation dataset."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        # Delete the individual tag item
        self._table.delete_item(pk=pk, sk=f"{SK_DATASET_TAG_PREFIX}{key}")
        # Remove from denormalized tags map on META
        self._remove_denormalized_tag(pk=pk, sk=SK_DATASET_META, tag_key=key)
        # Update last_update_time and digest
        now_ms = int(time.time() * 1000)
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is not None:
            digest = self._compute_dataset_digest(meta["name"], now_ms)
            self._table.update_item(
                pk=pk,
                sk=SK_DATASET_META,
                updates={"last_update_time": now_ms, "digest": digest},
            )

    def add_dataset_to_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> EvaluationDataset:
        """Associate an evaluation dataset with one or more experiments."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        for exp_id in experiment_ids:
            # put_item overwrites = idempotent
            self._table.put_item(
                {
                    "PK": pk,
                    "SK": f"{SK_DATASET_EXP_PREFIX}{exp_id}",
                    GSI1_PK: f"{GSI1_DS_EXP_PREFIX}{exp_id}",
                    GSI1_SK: dataset_id,
                }
            )
        # Update last_update_time and digest
        now_ms = int(time.time() * 1000)
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is not None:
            digest = self._compute_dataset_digest(meta["name"], now_ms)
            self._table.update_item(
                pk=pk,
                sk=SK_DATASET_META,
                updates={"last_update_time": now_ms, "digest": digest},
            )
        return self.get_dataset(dataset_id)

    def remove_dataset_from_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> EvaluationDataset:
        """Remove an evaluation dataset's association from one or more experiments."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        for exp_id in experiment_ids:
            self._table.delete_item(pk=pk, sk=f"{SK_DATASET_EXP_PREFIX}{exp_id}")
        # Update last_update_time and digest
        now_ms = int(time.time() * 1000)
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is not None:
            digest = self._compute_dataset_digest(meta["name"], now_ms)
            self._table.update_item(
                pk=pk,
                sk=SK_DATASET_META,
                updates={"last_update_time": now_ms, "digest": digest},
            )
        return self.get_dataset(dataset_id)

    # ------------------------------------------------------------------
    # Dataset Record CRUD
    # ------------------------------------------------------------------

    def upsert_dataset_records(
        self,
        dataset_id: str,
        records: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Upsert records into an evaluation dataset.

        Each record is keyed by its input_hash (SHA-256 of sorted JSON inputs).
        Existing records with the same input_hash are updated; new ones are
        inserted with a fresh ``edrec_<ulid>`` ID.

        Args:
            dataset_id: The ID of the dataset to update.
            records: List of record dicts with keys: inputs, outputs,
                expectations, tags, source.

        Returns:
            Dictionary with ``inserted`` and ``updated`` counts.
        """
        import json as _json

        from boto3.dynamodb.conditions import Attr

        # Verify dataset exists
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is None:
            raise MlflowException(
                f"Dataset '{dataset_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        inserted = 0
        updated = 0

        for record in records:
            inputs = record.get("inputs", {})
            outputs = record.get("outputs")
            expectations = record.get("expectations")
            record_tags = record.get("tags")
            source = record.get("source")

            # Compute input_hash
            input_hash = hashlib.sha256(
                _json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:8]

            # Dedup: query LSI3 for existing record with same input_hash
            existing_items, _ = self._table.query_page(
                pk=pk,
                sk_prefix=input_hash,
                index_name="lsi3",
                filter_expression=Attr("SK").begins_with(SK_DATASET_RECORD_PREFIX),
            )

            now_ms = int(time.time() * 1000)

            if existing_items:
                # Update the first matching record
                existing = existing_items[0]
                updates: dict[str, Any] = {
                    "last_update_time": now_ms,
                    LSI2_SK: now_ms,
                }
                if outputs is not None:
                    updates["outputs"] = outputs
                if expectations is not None:
                    updates["expectations"] = expectations
                if record_tags is not None:
                    updates["tags"] = record_tags
                if source is not None:
                    updates["source"] = source
                self._table.update_item(
                    pk=existing["PK"],
                    sk=existing["SK"],
                    updates=updates,
                )
                updated += 1
            else:
                # Insert new record
                record_id = f"edrec_{generate_ulid()}"
                record_item: dict[str, Any] = {
                    "PK": pk,
                    "SK": f"{SK_DATASET_RECORD_PREFIX}{record_id}",
                    "dataset_id": dataset_id,
                    "dataset_record_id": record_id,
                    "inputs": inputs,
                    "input_hash": input_hash,
                    "created_time": now_ms,
                    "last_update_time": now_ms,
                    # LSI projections
                    LSI1_SK: f"{now_ms:020d}",
                    LSI2_SK: now_ms,
                    LSI3_SK: input_hash,
                }
                if outputs is not None:
                    record_item["outputs"] = outputs
                if expectations is not None:
                    record_item["expectations"] = expectations
                if record_tags is not None:
                    record_item["tags"] = record_tags
                if source is not None:
                    record_item["source"] = source
                self._table.put_item(record_item)
                inserted += 1

        # Recount records to update META profile
        all_records, _ = self._table.query_page(
            pk=pk,
            sk_prefix=SK_DATASET_RECORD_PREFIX,
            limit=10000,
        )
        num_records = len(all_records)

        now_ms = int(time.time() * 1000)
        digest = self._compute_dataset_digest(meta["name"], now_ms)
        self._table.update_item(
            pk=pk,
            sk=SK_DATASET_META,
            updates={
                "profile": _json.dumps({"num_records": num_records}),
                "last_update_time": now_ms,
                "digest": digest,
                LSI2_SK: now_ms,
            },
        )

        return {"inserted": inserted, "updated": updated}

    def _load_dataset_records(
        self,
        dataset_id: str,
        max_results: int = 1000,
        page_token: str | None = None,
    ) -> tuple[list[Any], str | None]:
        """Load records for a dataset with pagination.

        Args:
            dataset_id: The dataset to load records for.
            max_results: Maximum records per page.
            page_token: Opaque cursor from a previous call.

        Returns:
            Tuple of (records, next_page_token). next_page_token is None on
            the last page.
        """
        from mlflow.entities.dataset_record import DatasetRecord

        from mlflow_dynamodbstore.dynamodb.pagination import (
            decode_page_token,
            encode_page_token,
        )

        pk = f"{PK_DATASET_PREFIX}{dataset_id}"

        token_state = decode_page_token(page_token)
        exclusive_start_key = token_state.get("lek") if token_state else None

        items, lek = self._table.query_page(
            pk=pk,
            sk_prefix=SK_DATASET_RECORD_PREFIX,
            limit=max_results,
            exclusive_start_key=exclusive_start_key,
        )

        record_list = [
            DatasetRecord(
                dataset_id=dataset_id,
                dataset_record_id=item["dataset_record_id"],
                inputs=item.get("inputs", {}),
                created_time=int(item["created_time"]),
                last_update_time=int(item["last_update_time"]),
                outputs=item.get("outputs"),
                expectations=item.get("expectations"),
                tags=item.get("tags"),
            )
            for item in items
        ]

        next_token = encode_page_token({"lek": lek}) if lek else None
        return record_list, next_token

    def delete_dataset_records(
        self,
        dataset_id: str,
        record_ids: list[str],
    ) -> int:
        """Delete specific records from a dataset.

        Args:
            dataset_id: The dataset to delete records from.
            record_ids: List of record IDs (``edrec_...``) to delete.

        Returns:
            Count of records deleted.
        """
        import json as _json

        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        keys = [{"PK": pk, "SK": f"{SK_DATASET_RECORD_PREFIX}{rec_id}"} for rec_id in record_ids]
        self._table.batch_delete(keys)

        # Recount and update META profile
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is not None:
            all_records, _ = self._table.query_page(
                pk=pk,
                sk_prefix=SK_DATASET_RECORD_PREFIX,
                limit=10000,
            )
            num_records = len(all_records)
            now_ms = int(time.time() * 1000)
            digest = self._compute_dataset_digest(meta["name"], now_ms)
            self._table.update_item(
                pk=pk,
                sk=SK_DATASET_META,
                updates={
                    "profile": _json.dumps({"num_records": num_records}),
                    "last_update_time": now_ms,
                    "digest": digest,
                    LSI2_SK: now_ms,
                },
            )

        return len(record_ids)
