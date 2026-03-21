"""DynamoDB-backed MLflow tracking store."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.entities.model_registry import PromptVersion
    from mlflow.entities.trace import Trace
    from mlflow.entities.trace_metrics import MetricAggregation, MetricDataPoint, MetricViewType
    from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig

    from mlflow_dynamodbstore.xray.client import XRayClient

from mlflow.entities import (
    Assessment,
    Dataset,
    DatasetInput,
    EvaluationDataset,
    Experiment,
    ExperimentTag,
    FallbackConfig,
    FallbackStrategy,
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelConfig,
    GatewayEndpointModelMapping,
    GatewayEndpointTag,
    GatewayModelDefinition,
    GatewayModelLinkageType,
    GatewayResourceType,
    GatewaySecretInfo,
    InputTag,
    LoggedModel,
    LoggedModelOutput,
    LoggedModelParameter,
    LoggedModelTag,
    Metric,
    Param,
    RoutingStrategy,
    Run,
    RunData,
    RunInfo,
    RunInputs,
    RunOutputs,
    RunTag,
    ScorerVersion,
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
    ViewType,
)
from mlflow.entities.dataset_summary import _DatasetSummary
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.analysis import TraceFilterCorrelationResult
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from mlflow.utils.crypto import KEKManager, _encrypt_secret, _mask_secret_value
from mlflow.utils.mlflow_tags import (
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_IS_GATEWAY,
    MLFLOW_EXPERIMENT_SOURCE_ID,
    MLFLOW_EXPERIMENT_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.utils.proto_json_utils import milliseconds_to_proto_timestamp
from mlflow.utils.time import get_current_time_millis
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
    GSI1_GW_SECRET_NAME_PREFIX,
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
    GSI2_GW_BIND_PREFIX,
    GSI2_GW_ENDPOINTS_PREFIX,
    GSI2_GW_MODELDEFS_PREFIX,
    GSI2_GW_SECRETS_PREFIX,
    GSI2_PK,
    GSI2_SESSIONS_PREFIX,
    GSI2_SK,
    GSI3_DS_NAME_PREFIX,
    GSI3_EXP_NAME_PREFIX,
    GSI3_GW_ENDPOINT_NAME_PREFIX,
    GSI3_GW_MODELDEF_NAME_PREFIX,
    GSI3_PK,
    GSI3_SCOR_NAME_PREFIX,
    GSI3_SK,
    GSI4_GW_MODELDEF_SECRET_PREFIX,
    GSI5_EXP_NAMES_PREFIX,
    GSI5_GW_ENDPOINT_MODELDEF_PREFIX,
    GSI5_PK,
    GSI5_SK,
    LSI1_SK,
    LSI2_SK,
    LSI3_SK,
    LSI4_SK,
    LSI5_SK,
    PK_DATASET_PREFIX,
    PK_EXPERIMENT_PREFIX,
    PK_GW_ENDPOINT_PREFIX,
    PK_GW_MODELDEF_PREFIX,
    PK_GW_SECRET_PREFIX,
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
    SK_GW_BIND_PREFIX,
    SK_GW_MAP_PREFIX,
    SK_GW_META,
    SK_GW_TAG_PREFIX,
    SK_INPUT_PREFIX,
    SK_INPUT_TAG_SUFFIX,
    SK_LM_METRIC_PREFIX,
    SK_LM_PARAM_PREFIX,
    SK_LM_PREFIX,
    SK_LM_TAG_PREFIX,
    SK_METRIC_HISTORY_PREFIX,
    SK_METRIC_PREFIX,
    SK_OUTPUT_PREFIX,
    SK_PARAM_PREFIX,
    SK_RANK_LM_PREFIX,
    SK_RANK_LMD_PREFIX,
    SK_RANK_PREFIX,
    SK_RUN_PREFIX,
    SK_SCORER_OSCFG_SUFFIX,
    SK_SCORER_PREFIX,
    SK_SESSION_PREFIX,
    SK_SPAN_METRIC_PREFIX,
    SK_SPAN_PREFIX,
    SK_TAG_PREFIX,
    SK_TRACE_METRIC_PREFIX,
    SK_TRACE_PREFIX,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri
from mlflow_dynamodbstore.ids import generate_ulid, ulid_from_timestamp

_logger = logging.getLogger(__name__)


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
    output_items: list[dict[str, Any]] | None = None,
) -> Run:
    """Convert DynamoDB items to an MLflow Run entity."""
    info = _item_to_run_info(item)
    # Populate run_name from mlflow.runName tag if not set on the item
    if not info.run_name:
        for t in tags:
            if t["key"] == "mlflow.runName":
                info._set_run_name(t["value"])
                break
    data = RunData(
        tags=[RunTag(t["key"], t["value"]) for t in tags],
        params=[Param(p["key"], p["value"]) for p in params],
        metrics=[
            Metric(m["key"], float(m["value"]), int(m.get("timestamp", 0)), int(m.get("step", 0)))
            for m in metrics
        ],
    )

    # Build RunInputs from input/dataset/input-tag items
    run_inputs = RunInputs(dataset_inputs=[])
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

    # Build RunOutputs from output items
    run_outputs = None
    if output_items:
        run_outputs = RunOutputs(
            model_outputs=[
                LoggedModelOutput(
                    model_id=o["destination_id"],
                    step=int(o.get("step", 0)),
                )
                for o in output_items
            ]
        )

    return Run(run_info=info, run_data=data, run_inputs=run_inputs, run_outputs=run_outputs)


def _item_to_logged_model(
    item: dict[str, Any],
    tags: list[dict[str, Any]] | None = None,
    params: list[dict[str, Any]] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> LoggedModel:
    """Convert DynamoDB items to a LoggedModel entity."""
    tag_dict = {t["key"]: t["value"] for t in (tags or [])}
    param_dict = {p["key"]: p["value"] for p in (params or [])}
    metric_list: list[Metric] | None = None
    if metrics:
        metric_list = [
            Metric(
                key=m["metric_name"],
                value=float(m["metric_value"]),
                timestamp=int(m.get("metric_timestamp_ms", 0)),
                step=int(m.get("metric_step", 0)),
                model_id=m.get("model_id"),
                dataset_name=m.get("dataset_name"),
                dataset_digest=m.get("dataset_digest"),
                run_id=m.get("run_id"),
            )
            for m in metrics
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


def _meta_to_dataset(dataset_id: str, meta: dict[str, Any]) -> EvaluationDataset:
    """Convert a DynamoDB dataset META item to an EvaluationDataset entity."""
    return EvaluationDataset(
        dataset_id=dataset_id,
        name=meta["name"],
        digest=meta["digest"],
        created_time=int(meta["created_time"]),
        last_update_time=int(meta["last_update_time"]),
        tags=meta.get("tags") or {},
        profile=meta.get("profile"),
        schema=meta.get("schema"),
        created_by=meta.get("created_by") or None,
        last_updated_by=meta.get("last_updated_by") or None,
    )


_UNSET = object()  # sentinel for optional parameters where None is a valid value


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
        self._cache = ResolutionCache(workspace=lambda: self._workspace)
        self._artifact_uri = artifact_uri or "./mlartifacts"
        self.artifact_root_uri = self._artifact_uri
        self._config = ConfigReader(self._table)
        self._config.reconcile()
        super().__init__()

    @property
    def _workspace(self) -> str:
        """Return the active workspace from context, defaulting to 'default'."""
        from mlflow.utils.workspace_context import get_request_workspace

        return get_request_workspace() or "default"

    @property
    def supports_workspaces(self) -> bool:
        """DynamoDB store always supports workspaces.

        Workspace scoping is built into the schema (GSI2/GSI3 prefixes,
        META workspace attribute). The --enable-workspaces server flag
        controls whether workspace features are active at runtime.
        """
        return True

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_max_results_param(self, max_results: int, allow_null: bool = False) -> None:
        from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD

        if (not allow_null and max_results is None) or (
            max_results is not None and max_results < 1
        ):
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                "a positive integer",
                INVALID_PARAMETER_VALUE,
            )
        if max_results is not None and max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                f"at most {SEARCH_MAX_RESULTS_THRESHOLD}",
                INVALID_PARAMETER_VALUE,
            )

    def _check_run_is_active(self, run_id: str) -> None:
        from mlflow.entities import LifecycleStage

        run = self.get_run(run_id)
        if run.info.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                f"The run {run_id} must be in the 'active' state. "
                f"Current state is {run.info.lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            )

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def _check_experiment_workspace(self, experiment_id: str) -> None:
        """Raise if the experiment does not belong to the current workspace."""
        item = self._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_META,
        )
        if item is None:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        item_workspace = item.get("workspace", "default")
        if item_workspace != self._workspace:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

    def _experiment_in_workspace(self, experiment_id: str) -> bool:
        """Return True if the experiment belongs to the current workspace."""
        item = self._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=SK_EXPERIMENT_META,
        )
        if item is None:
            return False
        return str(item.get("workspace", "default")) == self._workspace

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
            "artifact_location": artifact_location
            or append_to_uri_path(self._artifact_uri, str(exp_id)),
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
                f"No Experiment with id={experiment_id} exists",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        item_workspace = item.get("workspace", "default")
        if item_workspace != self._workspace:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists",
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
        levels = ("W", "3", "2")  # always trigram for experiment_name
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
                        forward_sk = f"{SK_FTS_PREFIX}{lvl}#E#{tok}#{experiment_id}"
                        self._table.delete_item(pk=pk, sk=forward_sk)
                        self._table.delete_item(pk=pk, sk=rev_sk)

        # Write new FTS items for added tokens
        if tokens_to_add:
            new_fts_items: list[dict[str, Any]] = []
            for lvl, tok in tokens_to_add:
                forward_sk = f"{SK_FTS_PREFIX}{lvl}#E#{tok}#{experiment_id}"
                reverse_sk = f"{SK_FTS_REV_PREFIX}E#{experiment_id}#{lvl}#{tok}"
                gsi2pk_val = f"{GSI2_FTS_NAMES_PREFIX}{self._workspace}"
                gsi2sk_val = f"{lvl}#E#{tok}#{experiment_id}"
                new_fts_items.append(
                    {"PK": pk, "SK": forward_sk, GSI2_PK: gsi2pk_val, GSI2_SK: gsi2sk_val}
                )
                new_fts_items.append({"PK": pk, "SK": reverse_sk})
            self._table.batch_write(new_fts_items)

        # Invalidate cache
        self._cache.invalidate("exp_name", old_name)

    def delete_experiment(self, experiment_id: str) -> None:
        """Soft-delete an experiment and set TTL on META only."""
        self._check_experiment_workspace(experiment_id)
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

        # Cascade: soft-delete all active runs in this experiment
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        active_runs = self._table.query(pk=pk, sk_prefix="active#", index_name="lsi1")
        for item in active_runs:
            sk = item.get("SK", "")
            if not sk.startswith(SK_RUN_PREFIX):
                continue
            run_id = sk[len(SK_RUN_PREFIX) :].split("#")[0]
            run_updates: dict[str, Any] = {
                "lifecycle_stage": "deleted",
                "deleted_time": now_ms,
                LSI1_SK: f"deleted#{run_id}",
            }
            if ttl_seconds is not None:
                run_updates["ttl"] = int(time.time()) + ttl_seconds
            self._table.update_item(pk=pk, sk=sk, updates=run_updates)

    def restore_experiment(self, experiment_id: str) -> None:
        """Restore a soft-deleted experiment and remove TTL from META."""
        self._check_experiment_workspace(experiment_id)
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

        # Cascade: restore all deleted runs in this experiment
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        deleted_runs = self._table.query(pk=pk, sk_prefix="deleted#", index_name="lsi1")
        for item in deleted_runs:
            sk = item.get("SK", "")
            if not sk.startswith(SK_RUN_PREFIX):
                continue
            run_id = sk[len(SK_RUN_PREFIX) :].split("#")[0]
            self._table.update_item(
                pk=pk,
                sk=sk,
                updates={
                    "lifecycle_stage": "active",
                    LSI1_SK: f"active#{run_id}",
                },
                removes=["ttl", "deleted_time"],
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
        self._validate_max_results_param(max_results)
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
                    sk_prefix=f"W#E#{token}#",
                    index_name="gsi2",
                )
                ids = set()
                for item in fts_items:
                    gsi2sk = item.get(GSI2_SK, "")
                    # Pattern: W#E#<token>#<exp_id>
                    parts = gsi2sk.split("#")
                    if len(parts) >= 4:
                        ids.add(parts[3])
                exp_ids = ids if exp_ids is None else exp_ids & ids
        else:
            # Fallback to trigrams
            trigram_tokens = tokenize_trigrams(substring)
            for token in trigram_tokens:
                fts_items = self._table.query(
                    pk=f"{GSI2_FTS_NAMES_PREFIX}{self._workspace}",
                    sk_prefix=f"3#E#{token}#",
                    index_name="gsi2",
                )
                ids = set()
                for item in fts_items:
                    gsi2sk = item.get(GSI2_SK, "")
                    # Pattern: 3#E#<token>#<exp_id>
                    parts = gsi2sk.split("#")
                    if len(parts) >= 4:
                        ids.add(parts[3])
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
        from mlflow.utils.validation import _validate_experiment_tag

        _validate_experiment_tag(tag.key, tag.value)
        # Verify experiment exists and is active
        exp = self.get_experiment(experiment_id)
        if exp.lifecycle_stage != "active":
            raise MlflowException(
                f"The experiment {experiment_id} must be in the 'active' state. "
                f"Current state is {exp.lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            )
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

        # Resolve run_name: use tag value, generate random name, or keep provided
        from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
        from mlflow.utils.name_utils import _generate_random_name

        run_name_tag = next((t for t in (tags or []) if t.key == MLFLOW_RUN_NAME), None)
        if run_name and run_name_tag and run_name != run_name_tag.value:
            raise MlflowException(
                f"Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                f"different values (run_name='{run_name}', "
                f"mlflow.runName='{run_name_tag.value}').",
                error_code=INVALID_PARAMETER_VALUE,
            )
        run_name = (
            run_name or (run_name_tag.value if run_name_tag else None) or _generate_random_name()
        )

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

        # Build the full tag list for the returned Run entity
        all_tags: list[RunTag] = list(tags or [])

        # Write mlflow.runName system tag
        if run_name and not run_name_tag:
            run_name_sys_tag = RunTag(MLFLOW_RUN_NAME, run_name)
            self._write_run_tag(experiment_id, run_id, run_name_sys_tag)
            all_tags.append(run_name_sys_tag)

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

        return self._build_run(item, all_tags)

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

        # Query output items
        output_items = self._table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk_prefix=f"{run_prefix}{SK_OUTPUT_PREFIX}",
        )

        return _item_to_run(
            item,
            tag_items,
            param_items,
            metric_items,
            input_items=input_items,
            dataset_items=dataset_items,
            input_tag_items=input_tag_items,
            output_items=output_items,
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
                f"Logged model with ID '{model_id}' not found.",
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
        from mlflow.utils.validation import _validate_logged_model_name

        _validate_logged_model_name(name)
        exp = self.get_experiment(experiment_id)
        if exp.lifecycle_stage != "active":
            raise MlflowException(
                f"The experiment {experiment_id} must be in the 'active' state. "
                f"Current state is {exp.lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            )

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
                f"Logged model with ID '{model_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        if meta.get("lifecycle_stage") == "deleted" and not allow_deleted:
            raise MlflowException(
                f"Logged model with ID '{model_id}' not found.",
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

    def log_logged_model_params(self, model_id: str, params: list[LoggedModelParameter]) -> None:
        """Log parameters on an existing logged model."""
        experiment_id = self._resolve_logged_model_experiment(model_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        for param in params:
            self._table.put_item(
                {
                    "PK": pk,
                    "SK": f"{SK_LM_PREFIX}{model_id}{SK_LM_PARAM_PREFIX}{param.key}",
                    "key": param.key,
                    "value": param.value,
                }
            )

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

        tag_sk = f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}{key}"
        existing = self._table.get_item(pk=pk, sk=tag_sk)
        if existing is None:
            raise MlflowException(
                f"No tag with key '{key}' for logged model '{model_id}'.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        self._table.delete_item(pk=pk, sk=tag_sk)

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

        # Validate filter string using MLflow's parser (raises with standard error message)
        if filter_string:
            from mlflow.utils.search_logged_model_utils import parse_filter_string

            parse_filter_string(filter_string)

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

        # Default sort: creation_timestamp descending
        models.sort(key=lambda m: m.creation_timestamp, reverse=True)

        # Validate and apply offset-based pagination
        sorted_exp_ids = sorted(experiment_ids)
        order_by_key = json.dumps(order_by, sort_keys=True) if order_by else ""
        filter_key = filter_string or ""

        token_data = decode_page_token(page_token)
        offset = 0
        if token_data:
            if token_data.get("experiment_ids") != sorted_exp_ids:
                raise MlflowException(
                    "Experiment IDs in the page token do not match the requested experiment IDs.",
                    INVALID_PARAMETER_VALUE,
                )
            if token_data.get("order_by", "") != order_by_key:
                raise MlflowException(
                    "Order by in the page token does not match the requested order by.",
                    INVALID_PARAMETER_VALUE,
                )
            if token_data.get("filter_string", "") != filter_key:
                raise MlflowException(
                    "Filter string in the page token does not match the requested filter string.",
                    INVALID_PARAMETER_VALUE,
                )
            offset = token_data.get("offset", 0)

        page = models[offset : offset + max_results]
        has_more = len(models) > offset + max_results
        next_token = (
            encode_page_token(
                {
                    "offset": offset + max_results,
                    "experiment_ids": sorted_exp_ids,
                    "order_by": order_by_key,
                    "filter_string": filter_key,
                }
            )
            if has_more
            else None
        )

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

        updates: dict[str, Any] = {}
        removes: list[str] = []
        if run_status is not None:
            updates["status"] = run_status
            updates[LSI3_SK] = f"{run_status}#{run_id}"
        if run_name:
            updates["run_name"] = run_name
            updates[LSI4_SK] = run_name.lower()

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

        # Update mlflow.runName tag when name changes
        old_run_name = current.get("run_name", "")
        if run_name and run_name != old_run_name:
            from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

            self._write_run_tag(experiment_id, run_id, RunTag(MLFLOW_RUN_NAME, run_name))

        # Update FTS for run name if it changed
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
            "deleted_time": get_current_time_millis(),
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

        # Update run META: restore lifecycle, remove TTL and deleted_time
        self._table.update_item(
            pk=pk,
            sk=f"{SK_RUN_PREFIX}{run_id}",
            updates={
                "lifecycle_stage": "active",
                LSI1_SK: f"active#{run_id}",
            },
            removes=["ttl", "deleted_time"],
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

    def _get_deleted_runs(self, older_than: int = 0) -> list[str]:
        """Return IDs of all soft-deleted runs across experiments in this workspace."""
        current_time = get_current_time_millis()
        # Get all experiments (active + deleted) in the workspace
        all_experiments = self.search_experiments(view_type=ViewType.ALL, max_results=50000)
        deleted_run_ids: list[str] = []
        for exp in all_experiments:
            pk = f"{PK_EXPERIMENT_PREFIX}{exp.experiment_id}"
            # LSI1 SK for deleted runs: "deleted#<run_id>"
            items = self._table.query(pk=pk, sk_prefix="deleted#", index_name="lsi1")
            for item in items:
                # Only include runs (SK starts with R#), not experiments
                if not item.get("SK", "").startswith(SK_RUN_PREFIX):
                    continue
                # Filter by deleted_time if older_than > 0
                if older_than > 0:
                    deleted_time = int(item.get("deleted_time", 0))
                    if deleted_time > current_time - older_than:
                        continue
                run_id = item.get("run_id") or item["SK"][len(SK_RUN_PREFIX) :].split("#")[0]
                deleted_run_ids.append(run_id)
        return deleted_run_ids

    def _get_deleted_logged_models(self, older_than: int = 0) -> list[str]:
        """Return IDs of all soft-deleted logged models across experiments."""
        current_time = get_current_time_millis()
        all_experiments = self.search_experiments(view_type=ViewType.ALL, max_results=50000)
        deleted_model_ids: list[str] = []
        for exp in all_experiments:
            pk = f"{PK_EXPERIMENT_PREFIX}{exp.experiment_id}"
            # Query all logged model META items
            items = self._table.query(pk=pk, sk_prefix=SK_LM_PREFIX)
            for item in items:
                # Only META items (SK = LM#<model_id>, no further # suffix)
                sk = item.get("SK", "")
                if sk.count("#") != 1:
                    continue
                if item.get("lifecycle_stage") != "deleted":
                    continue
                if older_than > 0:
                    last_updated = int(item.get("last_updated_timestamp_ms", 0))
                    if last_updated > current_time - older_than:
                        continue
                model_id = item.get("model_id") or sk[len(SK_LM_PREFIX) :]
                deleted_model_ids.append(model_id)
        return deleted_model_ids

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

        # 0. Filter experiment_ids to current workspace
        experiment_ids = [eid for eid in experiment_ids if self._experiment_in_workspace(eid)]

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
                    # Fetch dataset inputs for the run (INPUT links + ITAG children)
                    all_input_items = self._table.query(
                        pk=pk, sk_prefix=f"{run_prefix}{SK_INPUT_PREFIX}"
                    )
                    # Separate INPUT link items from ITAG items
                    input_items: list[dict[str, Any]] = []
                    input_tag_items: list[dict[str, Any]] = []
                    for it in all_input_items:
                        if SK_INPUT_TAG_SUFFIX in it["SK"]:
                            input_tag_items.append(it)
                        else:
                            input_items.append(it)
                    # Fetch dataset definitions (D# items in experiment partition)
                    dataset_items = (
                        self._table.query(pk=pk, sk_prefix=SK_DATASET_PREFIX) if input_items else []
                    )
                    # Fetch output items
                    output_items = self._table.query(
                        pk=pk, sk_prefix=f"{run_prefix}{SK_OUTPUT_PREFIX}"
                    )
                    runs.append(
                        _item_to_run(
                            item,
                            tag_items,
                            param_items,
                            metric_items,
                            input_items=input_items,
                            dataset_items=dataset_items,
                            input_tag_items=input_tag_items,
                            output_items=output_items,
                        )
                    )
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
        """Resolve run_id to experiment_id, enforcing workspace isolation.

        Uses cache then GSI1. Verifies the parent experiment belongs to the
        current workspace, raising with a run-specific message if not.
        """
        cached = self._cache.get("run_exp", run_id)
        if cached:
            if not self._experiment_in_workspace(cached):
                raise MlflowException(
                    f"Run with id={run_id} not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return cached

        # Look up via GSI1
        results = self._table.query(
            pk=f"{GSI1_RUN_PREFIX}{run_id}",
            index_name="gsi1",
            limit=1,
        )
        if not results:
            raise MlflowException(
                f"Run with id={run_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        experiment_id: str = results[0]["experiment_id"]
        if not self._experiment_in_workspace(experiment_id):
            raise MlflowException(
                f"Run with id={run_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
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
        # Sync run_name field when mlflow.runName tag is written
        from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

        if tag.key == MLFLOW_RUN_NAME and tag.value:
            updates: dict[str, Any] = {"run_name": tag.value, LSI4_SK: tag.value.lower()}
            self._table.update_item(pk=pk, sk=f"{SK_RUN_PREFIX}{run_id}", updates=updates)
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
        return Run(run_info=info, run_data=data, run_inputs=RunInputs(dataset_inputs=[]))

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
                forward_sk = f"{SK_FTS_PREFIX}{lvl}#{entity_type}#{tok}#{entity_id}{field_suffix}"
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
        levels = ("W", "3", "2")
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
                        fwd = f"{SK_FTS_PREFIX}{lvl}#{entity_type}#{tok}#{entity_id}{field_suffix}"
                        self._table.delete_item(pk=pk, sk=fwd)
                        self._table.delete_item(pk=pk, sk=rev_sk)

        # Write new FTS items for added tokens
        if tokens_to_add:
            add_gsi2 = entity_type in ("E", "M") and workspace is not None
            new_fts_items: list[dict[str, Any]] = []
            for lvl, tok in tokens_to_add:
                forward_sk = f"{SK_FTS_PREFIX}{lvl}#{entity_type}#{tok}#{entity_id}{field_suffix}"
                reverse_sk = f"{SK_FTS_REV_PREFIX}{entity_prefix}#{lvl}#{tok}"
                forward: dict[str, Any] = {"PK": pk, "SK": forward_sk}
                if add_gsi2:
                    forward[GSI2_PK] = f"{GSI2_FTS_NAMES_PREFIX}{workspace}"
                    forward[GSI2_SK] = f"{lvl}#{entity_type}#{tok}#{entity_id}{field_suffix}"
                new_fts_items.append(forward)
                new_fts_items.append({"PK": pk, "SK": reverse_sk})
            self._table.batch_write(new_fts_items)

    def log_metric(self, run_id: str, metric: Metric) -> None:
        """Log a single metric, validating before batch."""
        from mlflow.utils.validation import _validate_metric

        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
        self.log_batch(run_id, metrics=[metric], params=[], tags=[])

    def log_batch(
        self,
        run_id: str,
        metrics: list[Metric],
        params: list[Param],
        tags: list[RunTag],
    ) -> None:
        """Log a batch of metrics, params, and tags for a run."""
        from mlflow.utils.validation import (
            _validate_batch_log_data,
            _validate_batch_log_limits,
            _validate_param_keys_unique,
            _validate_run_id,
        )

        _validate_run_id(run_id)
        metrics, params, tags = _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)
        self._check_run_is_active(run_id)

        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        # Check for param overwrite (existing params with different values)
        if params:
            non_matching: list[dict[str, str]] = []
            for param in params:
                param_sk = f"{SK_RUN_PREFIX}{run_id}{SK_PARAM_PREFIX}{param.key}"
                existing = self._table.get_item(pk=pk, sk=param_sk)
                if existing is not None and existing.get("value") != param.value:
                    non_matching.append(
                        {
                            "key": param.key,
                            "old_value": existing["value"],
                            "new_value": param.value,
                        }
                    )
            if non_matching:
                raise MlflowException(
                    "Changing param values is not allowed. Params were already"
                    f" logged='{non_matching}' for run ID='{run_id}'.",
                    INVALID_PARAMETER_VALUE,
                )

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

            # RANK item for param (truncate value to stay within 1024-byte SK limit)
            ddb_sk_max = 1024
            rank_prefix = f"{SK_RANK_PREFIX}p#{param.key}#"
            rank_suffix = f"#{run_id}"
            max_val_bytes = ddb_sk_max - len(rank_prefix.encode()) - len(rank_suffix.encode())
            rank_val = param.value
            if len(rank_val.encode()) > max_val_bytes:
                # Truncate by bytes (UTF-8 safe); client-side re-sort fixes ordering
                rank_val = rank_val.encode()[:max_val_bytes].decode("utf-8", errors="ignore")

            rank_sk = f"{rank_prefix}{rank_val}{rank_suffix}"
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

        # Route metrics with model_id to logged model metric storage
        for metric in metrics:
            model_id = getattr(metric, "model_id", None)
            if model_id:
                self._log_logged_model_metric(
                    experiment_id=experiment_id,
                    model_id=model_id,
                    metric_name=metric.key,
                    metric_value=float(metric.value),
                    metric_timestamp_ms=metric.timestamp,
                    metric_step=metric.step,
                    run_id=run_id,
                    dataset_name=getattr(metric, "dataset_name", None),
                    dataset_digest=getattr(metric, "dataset_digest", None),
                )

        # Write tags individually (uses put_item)
        for tag in tags:
            self._write_run_tag(experiment_id, run_id, tag)

    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]) -> None:
        """Associate logged model outputs with a run."""
        if not models:
            return

        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        # Verify run is active (not deleted)
        meta = self._table.get_item(pk=pk, sk=f"{SK_RUN_PREFIX}{run_id}")
        if meta is None:
            raise MlflowException(
                f"Run '{run_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if meta.get("lifecycle_stage") == "deleted":
            raise MlflowException(
                f"Run '{run_id}' is deleted.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        items: list[dict[str, Any]] = []
        for model in models:
            output_id = generate_ulid()
            items.append(
                {
                    "PK": pk,
                    "SK": f"{SK_RUN_PREFIX}{run_id}{SK_OUTPUT_PREFIX}{output_id}",
                    "source_type": "RUN_OUTPUT",
                    "source_id": run_id,
                    "destination_type": "MODEL_OUTPUT",
                    "destination_id": model.model_id,
                    "step": model.step,
                }
            )

        self._table.batch_write(items)

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

    def get_metric_history_bulk_interval_from_steps(
        self, run_id: str, metric_key: str, steps: list[int], max_results: int
    ) -> list[Any]:
        """Return metric history for specific steps, optimized for DynamoDB."""
        from mlflow.entities.metric import MetricWithRunId

        if not steps:
            return []

        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        prefix = f"{SK_RUN_PREFIX}{run_id}{SK_METRIC_HISTORY_PREFIX}{metric_key}#"

        steps_set = set(steps)
        items = self._table.query(pk=pk, sk_prefix=prefix)

        metrics = sorted(
            [
                Metric(
                    key=item["key"],
                    value=float(item["value"]),
                    timestamp=int(item.get("timestamp", 0)),
                    step=int(item.get("step", 0)),
                )
                for item in items
                if int(item.get("step", 0)) in steps_set
            ],
            key=lambda m: (m.step, m.timestamp),
        )[:max_results]

        return [MetricWithRunId(run_id=run_id, metric=m) for m in metrics]

    def set_tag(self, run_id: str, tag: RunTag) -> None:
        """Set a tag on a run."""
        from mlflow.utils.validation import _validate_tag

        tag = _validate_tag(tag.key, tag.value)
        self._check_run_is_active(run_id)
        experiment_id = self._resolve_run_experiment(run_id)
        self._write_run_tag(experiment_id, run_id, tag)

    def delete_tag(self, run_id: str, key: str) -> None:
        """Delete a tag from a run."""
        self._check_run_is_active(run_id)
        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_RUN_PREFIX}{run_id}{SK_TAG_PREFIX}{key}"
        existing = self._table.get_item(pk=pk, sk=sk)
        if existing is None:
            raise MlflowException(
                f"No tag with name: {key}",
                INVALID_PARAMETER_VALUE,
            )
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
        from mlflow.utils.validation import _validate_dataset_inputs

        if datasets is not None:
            if not isinstance(datasets, list):
                raise MlflowException(
                    f"Argument 'datasets' should be a list, got '{type(datasets)}'",
                    INVALID_PARAMETER_VALUE,
                )
            _validate_dataset_inputs(datasets)
        if not datasets:
            return

        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        items: list[dict[str, Any]] = []

        # Deduplicate datasets by (name, digest) to avoid duplicate keys in batch_write
        seen_datasets: set[tuple[str, str]] = set()
        for dataset_input in datasets:
            ds = dataset_input.dataset
            ds_key = (ds.name, ds.digest)
            if ds_key in seen_datasets:
                continue
            seen_datasets.add(ds_key)
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

    def _search_datasets(self, experiment_ids: list[str]) -> list[_DatasetSummary]:
        """Search for legacy V2 datasets (D# and DLINK# items) under experiment partitions.

        This method queries existing D# (dataset) and DLINK# (dataset-run link) items
        that are stored under experiment partitions when log_inputs is called.

        Args:
            experiment_ids: List of experiment IDs to search within.

        Returns:
            List of DatasetSummary protobuf objects.
        """
        # Filter experiment_ids to current workspace
        experiment_ids = [eid for eid in experiment_ids if self._experiment_in_workspace(eid)]

        # Collect unique (experiment_id, name, digest, context) summaries
        summaries: set[tuple[str, str, str, str | None]] = set()
        datasets: dict[tuple[str, str], dict[str, Any]] = {}

        for experiment_id in experiment_ids:
            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

            # Query D# items (dataset definitions)
            d_items = self._table.query(pk=pk, sk_prefix=SK_DATASET_PREFIX)
            for item in d_items:
                sk = item.get("SK", "")
                if sk.startswith(SK_DATASET_PREFIX):
                    parts = sk[len(SK_DATASET_PREFIX) :].split("#", 1)
                    if len(parts) == 2:
                        name, digest = parts
                        datasets[(name, digest)] = {
                            "experiment_id": experiment_id,
                            "name": item.get("name", name),
                            "digest": item.get("digest", digest),
                        }

            # Query DLINK# items for context — each unique context creates a summary
            dlink_items = self._table.query(pk=pk, sk_prefix=SK_DLINK_PREFIX)
            linked_datasets: set[tuple[str, str]] = set()
            for item in dlink_items:
                sk = item.get("SK", "")
                if sk.startswith(SK_DLINK_PREFIX):
                    parts = sk[len(SK_DLINK_PREFIX) :].split("#", 2)
                    if len(parts) >= 2:
                        name, digest = parts[0], parts[1]
                        context = item.get("context")
                        linked_datasets.add((name, digest))
                        summaries.add((experiment_id, name, digest, context))

            # Add datasets without any DLINK (no context)
            for (name, digest), data in datasets.items():
                if data["experiment_id"] == experiment_id and (name, digest) not in linked_datasets:
                    summaries.add((experiment_id, name, digest, None))

        results: list[_DatasetSummary] = [
            _DatasetSummary(experiment_id=eid, name=n, digest=d, context=c)
            for eid, n, d, c in summaries
        ]
        return results[:1000]

    # ------------------------------------------------------------------
    # Trace CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _get_trace_status_from_spans(spans: list[Any]) -> str | None:
        """Infer trace status from root span if present."""
        from mlflow.entities import SpanStatusCode, TraceState

        for span in spans:
            parent_id = getattr(span, "parent_id", None)
            if parent_id is None:
                status = getattr(span, "status", None)
                if status is not None:
                    code = getattr(status, "status_code", None)
                    if code == SpanStatusCode.ERROR:
                        return TraceState.ERROR.value
                    return TraceState.OK.value
        return None

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
                f"Trace with ID {trace_id} is not found.",
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

        try:
            self._table.put_item(item, condition="attribute_not_exists(PK)")
        except Exception:
            # Trace META may already exist (created by log_spans) — update instead
            update_fields = {k: v for k, v in item.items() if k not in ("PK", "SK")}
            self._table.update_item(pk=pk, sk=sk, updates=update_fields)

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
        session_id = (trace_info.trace_metadata or {}).get(TraceMetadataKey.TRACE_SESSION)
        if session_id:
            self._upsert_session_tracker(
                experiment_id=experiment_id,
                session_id=session_id,
                timestamp_ms=trace_info.request_time,
                ttl=ttl,
            )

        # Write assessments with backfilled trace_id
        if trace_info.assessments:
            for assessment in trace_info.assessments:
                if assessment.trace_id is None:
                    assessment.trace_id = trace_id
                self.create_assessment(assessment)
            # Reload to return persisted state
            return self.get_trace_info(trace_id)

        return trace_info

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        """Fetch a trace by ID, reconstructing TraceInfo from DynamoDB items."""
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}"

        meta = self._table.get_item(pk=pk, sk=sk)
        if meta is None:
            raise MlflowException(
                f"Trace with ID {trace_id} is not found.",
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

        assessments = self._load_assessments(pk, trace_id)

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
            assessments=assessments,
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

            # Denormalize span attributes on META (skip if already done)
            if span_dicts and not (meta or {}).get("span_types"):
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
        # Try V3 format (Span.to_dict) first — preserves original span IDs.
        # Fall back to X-Ray converter which hashes span IDs.
        if span_dicts and "start_time_unix_nano" in span_dicts[0]:
            try:
                from mlflow.entities.span import Span as SpanEntity

                spans = [SpanEntity.from_dict(sd) for sd in span_dicts]
            except Exception:
                spans = span_dicts_to_mlflow_spans(span_dicts, trace_id)
        else:
            spans = span_dicts_to_mlflow_spans(span_dicts, trace_id)

        # Check for partial trace when not allowed
        if not allow_partial and spans:
            import json as _json2

            size_stats_str = trace_info.trace_metadata.get(TraceMetadataKey.SIZE_STATS)
            if size_stats_str:
                from mlflow.tracing.constant import TraceSizeStatsKey

                size_stats = _json2.loads(size_stats_str)
                expected_spans = size_stats.get(TraceSizeStatsKey.NUM_SPANS, 0)
                if expected_spans > len(spans):
                    raise MlflowException(
                        f"Trace with ID {trace_id} is not fully exported yet. "
                        f"Expected {expected_spans} spans but only {len(spans)} are available.",
                        INVALID_STATE,
                    )

        return Trace(info=trace_info, data=TraceData(spans=spans))

    def log_spans(
        self, location: str, spans: list[Any], tracking_uri: str | None = None
    ) -> list[Any]:
        """Log spans to the tracking store by writing SPANS cache items.

        In addition to the SPANS JSON blob, writes:
        - Individual span items (T#<trace_id>#SPAN#<span_id>)
        - Trace metric items (T#<trace_id>#TMETRIC#<key>) for aggregated token usage
        - Span metric items (T#<trace_id>#SMETRIC#<span_id>#<key>) for per-span costs
        - META denormalization of span_types, span_names, span_statuses
        """
        import json as _json
        from collections import defaultdict

        if not spans:
            return []

        # Group spans by trace_id
        spans_by_trace: dict[str, list[Any]] = defaultdict(list)
        for span in spans:
            spans_by_trace[span.trace_id].append(span)

        for trace_id, trace_spans in spans_by_trace.items():
            try:
                experiment_id = location or self._resolve_trace_experiment(trace_id)
            except MlflowException:
                if not location:
                    continue  # Skip unresolvable traces
                experiment_id = location

            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            sk = f"{SK_TRACE_PREFIX}{trace_id}"

            # Read TTL from trace META, create if not exists
            meta = self._table.get_item(pk=pk, sk=sk)
            ttl = int(meta["ttl"]) if meta and "ttl" in meta else self._get_trace_ttl()

            if meta is None:
                # Trace doesn't exist yet — create META from span data
                from mlflow.entities import TraceState

                min_start_ns = min(
                    getattr(s, "start_time_ns", None) or s.to_dict().get("start_time_ns", 0)
                    for s in trace_spans
                )
                request_time = min_start_ns // 1_000_000

                end_times = [
                    getattr(s, "end_time_ns", None) or s.to_dict().get("end_time_ns")
                    for s in trace_spans
                ]
                end_times = [t for t in end_times if t is not None and t > 0]
                max_end_ms = (max(end_times) // 1_000_000) if end_times else None
                execution_duration = (max_end_ms - request_time) if max_end_ms else 0

                root_status = self._get_trace_status_from_spans(trace_spans)
                state_str = root_status or TraceState.IN_PROGRESS.value

                meta_item: dict[str, Any] = {
                    "PK": pk,
                    "SK": sk,
                    "trace_id": trace_id,
                    "experiment_id": experiment_id,
                    "request_time": request_time,
                    "execution_duration": execution_duration,
                    "state": state_str,
                    "tags": {},
                    LSI1_SK: f"{request_time:020d}",
                    LSI2_SK: request_time + execution_duration,
                    LSI3_SK: f"{state_str}#{request_time:020d}",
                    LSI5_SK: f"{execution_duration:020d}",
                    GSI1_PK: f"{GSI1_TRACE_PREFIX}{trace_id}",
                    GSI1_SK: f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
                }
                if ttl is not None:
                    meta_item["ttl"] = ttl
                self._table.put_item(meta_item)
                meta = meta_item

                # Cache trace_id -> experiment_id
                self._cache.put("trace_exp", trace_id, experiment_id)

            span_dicts = [s.to_dict() for s in trace_spans]
            spans_sk = f"{SK_TRACE_PREFIX}{trace_id}#SPANS"

            # Write spansLocation tag (indicates spans are stored in tracking store)
            from mlflow.tracing.constant import SpansLocation

            self._write_trace_tag(
                experiment_id,
                trace_id,
                TraceTagKey.SPANS_LOCATION,
                SpansLocation.TRACKING_STORE.value,
                ttl,
            )

            # --- Write individual span items, metrics, and denormalize META ---
            extra_items: list[dict[str, Any]] = []
            span_types: set[str] = set()
            span_statuses: set[str] = set()
            span_names: set[str] = set()
            # Accumulators for trace-level token usage and cost
            total_input_tokens = 0
            total_output_tokens = 0
            total_total_tokens = 0
            has_token_usage = False
            total_input_cost = 0.0
            total_output_cost = 0.0
            total_total_cost = 0.0
            has_cost = False
            session_id_from_spans: str | None = None

            for span in trace_spans:
                sd = span.to_dict()
                attrs = sd.get("attributes", {})

                # Read span fields — prefer direct properties, fall back to dict
                span_id = getattr(span, "span_id", None) or sd.get("span_id")
                name = getattr(span, "name", None) or sd.get("name", "")
                span_type = getattr(span, "span_type", None) or sd.get("span_type", "")
                start_ns = getattr(span, "start_time_ns", None)
                if start_ns is None or not isinstance(start_ns, int | float):
                    start_ns = sd.get("start_time_ns", sd.get("start_time_unix_nano", 0))
                end_ns = getattr(span, "end_time_ns", None)
                if end_ns is None or not isinstance(end_ns, int | float):
                    end_ns = sd.get("end_time_ns", sd.get("end_time_unix_nano", 0))

                # Extract status string
                status = getattr(span, "status", None) or sd.get("status", "")
                if hasattr(status, "status_code"):
                    status_str = str(status.status_code)
                elif isinstance(status, dict):
                    status_str = str(status.get("code", status))
                else:
                    status_str = str(status)

                # Collect for META denormalization
                if span_type:
                    span_types.add(str(span_type))
                if status_str:
                    span_statuses.add(status_str)
                if name:
                    span_names.add(str(name))

                # Skip individual span item if we don't have a span_id
                if not span_id:
                    continue

                # Build individual span item
                span_item: dict[str, Any] = {
                    "PK": pk,
                    "SK": f"{SK_TRACE_PREFIX}{trace_id}{SK_SPAN_PREFIX}{span_id}",
                    "name": str(name),
                    "type": str(span_type),
                    "status": status_str,
                    "start_time_ns": int(start_ns),
                    "end_time_ns": int(end_ns),
                    "duration_ms": (int(end_ns) - int(start_ns)) // 1_000_000,
                }
                model_name = attrs.get("mlflow.llm.model")
                if model_name:
                    try:
                        model_name = _json.loads(model_name)
                    except (TypeError, _json.JSONDecodeError):
                        pass
                    if model_name:
                        span_item["model_name"] = model_name

                model_provider = attrs.get("mlflow.llm.provider")
                if model_provider:
                    try:
                        model_provider = _json.loads(model_provider)
                    except (TypeError, _json.JSONDecodeError):
                        pass
                    if model_provider:
                        span_item["model_provider"] = model_provider

                # Extract session.id from span attributes (OTel semantic convention)
                if session_id_from_spans is None:
                    span_session = attrs.get("session.id")
                    if span_session:
                        session_id_from_spans = span_session

                if ttl is not None:
                    span_item["ttl"] = ttl
                extra_items.append(span_item)

                # --- Token usage (aggregate to trace level) ---
                token_usage_raw = attrs.get("mlflow.chat.tokenUsage")
                if token_usage_raw:
                    try:
                        token_usage = (
                            _json.loads(token_usage_raw)
                            if isinstance(token_usage_raw, str)
                            else token_usage_raw
                        )
                        total_input_tokens += int(token_usage.get("input_tokens", 0))
                        total_output_tokens += int(token_usage.get("output_tokens", 0))
                        total_total_tokens += int(token_usage.get("total_tokens", 0))
                        has_token_usage = True
                    except (TypeError, _json.JSONDecodeError, ValueError):
                        pass

                # --- Per-span cost metrics ---
                cost_raw = attrs.get("mlflow.llm.cost")
                if cost_raw:
                    try:
                        from decimal import Decimal as _Decimal

                        cost = _json.loads(cost_raw) if isinstance(cost_raw, str) else cost_raw
                        for cost_key in ("input_cost", "output_cost", "total_cost"):
                            if cost_key in cost and cost[cost_key] is not None:
                                cost_item: dict[str, Any] = {
                                    "PK": pk,
                                    "SK": (
                                        f"{SK_TRACE_PREFIX}{trace_id}"
                                        f"{SK_SPAN_METRIC_PREFIX}{span_id}#{cost_key}"
                                    ),
                                    "value": _Decimal(str(cost[cost_key])),
                                    "key": cost_key,
                                    "span_id": span_id,
                                }
                                if ttl is not None:
                                    cost_item["ttl"] = ttl
                                extra_items.append(cost_item)
                        # Accumulate trace-level cost
                        total_input_cost += float(cost.get("input_cost", 0) or 0)
                        total_output_cost += float(cost.get("output_cost", 0) or 0)
                        total_total_cost += float(cost.get("total_cost", 0) or 0)
                        has_cost = True
                    except (TypeError, _json.JSONDecodeError, ValueError):
                        pass

            # --- Write trace-level metric items ---
            if has_token_usage:
                from decimal import Decimal as _Decimal

                for metric_key, metric_val in [
                    ("input_tokens", total_input_tokens),
                    ("output_tokens", total_output_tokens),
                    ("total_tokens", total_total_tokens),
                ]:
                    tmetric_item: dict[str, Any] = {
                        "PK": pk,
                        "SK": (f"{SK_TRACE_PREFIX}{trace_id}{SK_TRACE_METRIC_PREFIX}{metric_key}"),
                        "value": _Decimal(str(metric_val)),
                        "key": metric_key,
                    }
                    if ttl is not None:
                        tmetric_item["ttl"] = ttl
                    extra_items.append(tmetric_item)

            # --- Write trace-level token_usage and cost as RMETA items ---
            # Accumulate with existing values (log_spans may be called multiple times)
            if has_token_usage:
                existing_token = self._table.get_item(
                    pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}#RMETA#mlflow.trace.tokenUsage"
                )
                if existing_token:
                    prev = _json.loads(existing_token["value"])
                    total_input_tokens += int(prev.get("input_tokens", 0))
                    total_output_tokens += int(prev.get("output_tokens", 0))
                    total_total_tokens += int(prev.get("total_tokens", 0))
                token_data = _json.dumps(
                    {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_total_tokens,
                    }
                )
                rmeta_token: dict[str, Any] = {
                    "PK": pk,
                    "SK": f"{SK_TRACE_PREFIX}{trace_id}#RMETA#mlflow.trace.tokenUsage",
                    "key": "mlflow.trace.tokenUsage",
                    "value": token_data,
                }
                if ttl is not None:
                    rmeta_token["ttl"] = ttl
                extra_items.append(rmeta_token)

            if has_cost:
                existing_cost = self._table.get_item(
                    pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}#RMETA#mlflow.trace.cost"
                )
                if existing_cost:
                    prev = _json.loads(existing_cost["value"])
                    total_input_cost += float(prev.get("input_cost", 0))
                    total_output_cost += float(prev.get("output_cost", 0))
                    total_total_cost += float(prev.get("total_cost", 0))
                cost_data = _json.dumps(
                    {
                        "input_cost": total_input_cost,
                        "output_cost": total_output_cost,
                        "total_cost": total_total_cost,
                    }
                )
                rmeta_cost: dict[str, Any] = {
                    "PK": pk,
                    "SK": f"{SK_TRACE_PREFIX}{trace_id}#RMETA#mlflow.trace.cost",
                    "key": "mlflow.trace.cost",
                    "value": cost_data,
                }
                if ttl is not None:
                    rmeta_cost["ttl"] = ttl
                extra_items.append(rmeta_cost)

            # Write all extra items in batch (individual span items, metrics, etc.)
            if extra_items:
                self._table.batch_write(extra_items)

            # Rebuild SPANS blob with optimistic locking (retry on conflict).
            # Each write increments a version counter; if another thread wrote
            # between our read and write, the condition fails and we retry.
            from botocore.exceptions import ClientError

            for _attempt in range(10):
                existing_blob = self._table.get_item(pk=pk, sk=spans_sk)
                if existing_blob is not None:
                    existing_dicts = _json.loads(existing_blob["data"])
                    merged = {sd.get("span_id"): sd for sd in existing_dicts}
                    old_version = int(existing_blob.get("version", 0))
                else:
                    merged = {}
                    old_version = 0
                for sd in span_dicts:
                    merged[sd.get("span_id")] = sd
                new_version = old_version + 1
                spans_item: dict[str, Any] = {
                    "PK": pk,
                    "SK": spans_sk,
                    "data": _json.dumps(list(merged.values())),
                    "version": new_version,
                }
                if ttl is not None:
                    spans_item["ttl"] = ttl
                try:
                    if old_version == 0:
                        self._table.put_item(spans_item, condition="attribute_not_exists(version)")
                    else:
                        from mlflow_dynamodbstore.dynamodb.table import convert_floats

                        self._table._table.put_item(
                            Item=convert_floats(spans_item),
                            ConditionExpression="version = :v",
                            ExpressionAttributeValues={":v": old_version},
                        )
                    break
                except ClientError as e:
                    if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                        continue  # Retry on conflict
                    raise

            # Write session ID from span attributes if not already set
            if session_id_from_spans:
                session_rmeta_sk = (
                    f"{SK_TRACE_PREFIX}{trace_id}#RMETA#{TraceMetadataKey.TRACE_SESSION}"
                )
                existing_session = self._table.get_item(pk=pk, sk=session_rmeta_sk)
                if existing_session is None:
                    rmeta_item: dict[str, Any] = {
                        "PK": pk,
                        "SK": session_rmeta_sk,
                        "key": TraceMetadataKey.TRACE_SESSION,
                        "value": session_id_from_spans,
                    }
                    if ttl is not None:
                        rmeta_item["ttl"] = ttl
                    self._table.put_item(rmeta_item)
                    # Also upsert session tracker
                    request_time = int(meta.get("request_time", 0))
                    self._upsert_session_tracker(
                        experiment_id, session_id_from_spans, request_time, ttl
                    )

            # --- Denormalize span attributes on META ---
            updates: dict[str, Any] = {}
            if span_types:
                updates["span_types"] = span_types
            if span_statuses:
                updates["span_statuses"] = span_statuses
            if span_names:
                updates["span_names"] = span_names

            # Update trace state from root span if status changed
            root_status = self._get_trace_status_from_spans(trace_spans)
            current_state = meta.get("state", "")
            if root_status and root_status != current_state:
                from mlflow.entities import TraceState

                finalized = {TraceState.OK.value, TraceState.ERROR.value}
                if current_state not in finalized:
                    updates["state"] = root_status
                    request_time = int(meta.get("request_time", 0))
                    updates[LSI3_SK] = f"{root_status}#{request_time:020d}"

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

        return spans

    async def log_spans_async(self, location: str, spans: list[Any]) -> list[Any]:
        """Async version of log_spans — delegates to synchronous implementation."""
        return self.log_spans(location, spans)

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
        self._validate_max_results_param(max_results)

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

        # 1. Parse filter (validates syntax including prompt filter format)
        predicates = parse_trace_filter(filter_string)

        # Validate prompt filter constraints
        for pred in predicates:
            if pred.key == TraceTagKey.LINKED_PROMPTS:
                if pred.op != "=":
                    raise MlflowException(
                        f"Invalid comparator '{pred.op}' for prompts filter. "
                        "Only '=' is supported with format: prompt = \"name/version\"",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if pred.value.count("/") != 1:
                    raise MlflowException(
                        f"Invalid prompts filter value '{pred.value}'. "
                        'Expected format: prompt = "name/version"',
                        error_code=INVALID_PARAMETER_VALUE,
                    )
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

    def calculate_trace_filter_correlation(
        self,
        experiment_ids: list[str],
        filter_string1: str,
        filter_string2: str,
        base_filter: str | None = None,
    ) -> TraceFilterCorrelationResult:
        """Calculate NPMI correlation between two trace filters."""

        from mlflow_dynamodbstore.dynamodb.search import (
            _apply_trace_post_filter,
            execute_trace_query,
            parse_trace_filter,
            plan_trace_query,
        )

        preds1 = parse_trace_filter(filter_string1)
        preds2 = parse_trace_filter(filter_string2)
        base_preds = parse_trace_filter(base_filter)

        # Plan query using base_filter predicates (for efficient index usage)
        plan = plan_trace_query(base_preds, None)

        total_count = 0
        filter1_count = 0
        filter2_count = 0
        joint_count = 0

        for exp_id in experiment_ids:
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
            page_token: str | None = None

            while True:
                items, page_token = execute_trace_query(
                    table=self._table,
                    plan=plan,
                    pk=pk,
                    max_results=1000,
                    page_token=page_token,
                    predicates=base_preds,
                )

                for item in items:
                    trace_id = item["trace_id"]
                    total_count += 1

                    match1 = all(
                        _apply_trace_post_filter(self._table, pk, trace_id, item, p) for p in preds1
                    )
                    match2 = all(
                        _apply_trace_post_filter(self._table, pk, trace_id, item, p) for p in preds2
                    )

                    if match1:
                        filter1_count += 1
                    if match2:
                        filter2_count += 1
                    if match1 and match2:
                        joint_count += 1

                if not page_token:
                    break

        # Compute NPMI using MLflow's standard implementation
        from mlflow.store.analytics.trace_correlation import calculate_npmi_from_counts

        npmi_result = calculate_npmi_from_counts(
            joint_count=joint_count,
            filter1_count=filter1_count,
            filter2_count=filter2_count,
            total_count=total_count,
        )

        return TraceFilterCorrelationResult(
            npmi=npmi_result.npmi,
            npmi_smoothed=npmi_result.npmi_smoothed,
            filter1_count=filter1_count,
            filter2_count=filter2_count,
            joint_count=joint_count,
            total_count=total_count,
        )

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
                # Unknown span key (e.g. "content" from trace.text ILIKE).
                # Can't filter from cached META sets — skip this predicate
                # and let the trace through rather than rejecting it.
                continue
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

        assessments = self._load_assessments(pk, trace_id)

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
            assessments=assessments,
        )

    def _load_assessments(self, pk: str, trace_id: str) -> list[Assessment]:
        """Load assessments for a trace, fixing the valid field for expectations."""
        assess_items = self._table.query(
            pk=pk,
            sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#",
        )
        assessments: list[Assessment] = []
        for item in assess_items:
            if "data" not in item:
                continue
            try:
                a = Assessment.from_dictionary(item["data"])
                # Expectation.to_dictionary() stores valid=None, but the SQL store
                # always sets valid=True. Proto3 omits None values, causing the UI
                # to not render the assessment. Fix by defaulting to True.
                if a.valid is None:
                    a.valid = True
                assessments.append(a)
            except Exception:
                pass
        return assessments

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
            levels = ("W", "3", "2")
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
                            forward_sk = f"{SK_FTS_PREFIX}{lvl}#T#{tok}#{trace_id}{field_suffix}"
                            self._table.delete_item(pk=pk, sk=forward_sk)
                            self._table.delete_item(pk=pk, sk=rev_sk)

            # Write new FTS items for added tokens
            if tokens_to_add:
                new_fts_items: list[dict[str, Any]] = []
                for lvl, tok in tokens_to_add:
                    forward_sk = f"{SK_FTS_PREFIX}{lvl}#T#{tok}#{trace_id}{field_suffix}"
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
        from mlflow.utils.validation import _validate_trace_tag

        key, value = _validate_trace_tag(key, value)
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        # Read TTL from the trace META item
        meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
        if meta is None:
            raise MlflowException(
                f"Trace with ID {trace_id} is not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        ttl = int(meta["ttl"]) if "ttl" in meta else self._get_trace_ttl()
        self._write_trace_tag(experiment_id, trace_id, key, value, ttl)

    def delete_trace_tag(self, trace_id: str, key: str) -> None:
        """Delete a tag from a trace."""
        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#TAG#{key}"
        existing = self._table.get_item(pk=pk, sk=sk)
        if existing is None:
            raise MlflowException(
                f"No trace tag with key '{key}'",
                INVALID_PARAMETER_VALUE,
            )
        self._table.delete_item(pk=pk, sk=sk)
        if self._config.should_denormalize(experiment_id, key):
            self._remove_denormalized_tag(pk, f"{SK_TRACE_PREFIX}{trace_id}", key)
        # Remove FTS items for tag value if FTS was configured
        if self._config.should_trigram("trace_tag_value"):
            self._delete_fts_for_entity_field(pk=pk, entity_type="T", entity_id=trace_id, field=key)

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        """Link traces to a run by writing mlflow.sourceRun request metadata."""
        max_trace_links = 100

        if not trace_ids:
            return
        if len(trace_ids) > max_trace_links:
            raise MlflowException(
                f"Cannot link more than {max_trace_links} traces to a run in "
                f"a single request. Provided {len(trace_ids)} traces.",
                error_code=INVALID_PARAMETER_VALUE,
            )
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
                f"Trace with ID {trace_id} is not found.",
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
                # Try V3 format (Span.to_dict) first, fall back to X-Ray format
                if span_dicts and "start_time_unix_nano" in span_dicts[0]:
                    try:
                        from mlflow.entities.span import Span as SpanEntity

                        spans = [SpanEntity.from_dict(sd) for sd in span_dicts]
                    except MlflowException:
                        spans = span_dicts_to_mlflow_spans(span_dicts, trace_id)
                else:
                    # X-Ray converter format
                    spans = span_dicts_to_mlflow_spans(span_dicts, trace_id)
            else:
                spans = []

            # Sort spans by start_time_ns ascending (JSON blob is unordered)
            spans.sort(key=lambda s: getattr(s, "start_time_ns", 0) or 0)

            # Skip incomplete traces (expected span count > actual)
            from mlflow.tracing.constant import TraceSizeStatsKey

            size_stats_str = trace_info.trace_metadata.get(TraceMetadataKey.SIZE_STATS)
            if size_stats_str:
                size_stats = _json.loads(size_stats_str)
                expected = size_stats.get(TraceSizeStatsKey.NUM_SPANS, 0)
                if expected > len(spans):
                    continue

            results.append(Trace(info=trace_info, data=TraceData(spans=spans)))

        return results

    def find_completed_sessions(
        self,
        experiment_id: str,
        min_last_trace_timestamp_ms: int,
        max_last_trace_timestamp_ms: int,
        max_results: int | None = None,
        filter_string: str | None = None,
    ) -> list[Any]:
        """Find completed sessions by last trace timestamp range via GSI2."""
        from mlflow.genai.scorers.online.entities import CompletedSession

        gsi2pk = f"{GSI2_SESSIONS_PREFIX}{self._workspace}#{experiment_id}"

        items = self._table.query(
            pk=gsi2pk,
            sk_gte=f"{min_last_trace_timestamp_ms:020d}",
            sk_lte=f"{max_last_trace_timestamp_ms:020d}",
            index_name="gsi2",
            scan_forward=True,
        )

        # Optional: post-filter sessions by trace attributes
        if filter_string:
            from mlflow_dynamodbstore.dynamodb.search import (
                _apply_trace_post_filter,
                parse_trace_filter,
            )

            preds = parse_trace_filter(filter_string)
            filtered_items = []
            for item in items:
                session_id = item["session_id"]
                exp_pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
                trace_items = self._table.query(
                    pk=exp_pk,
                    sk_prefix=SK_TRACE_PREFIX,
                )
                session_qualifies = False
                for t_item in trace_items:
                    if "trace_id" not in t_item:
                        continue
                    tid = t_item["trace_id"]
                    rmeta_sk = f"{SK_TRACE_PREFIX}{tid}#RMETA#{TraceMetadataKey.TRACE_SESSION}"
                    rmeta = self._table.get_item(pk=exp_pk, sk=rmeta_sk)
                    if rmeta and rmeta.get("value") == session_id:
                        if all(
                            _apply_trace_post_filter(self._table, exp_pk, tid, t_item, p)
                            for p in preds
                        ):
                            session_qualifies = True
                            break
                if session_qualifies:
                    filtered_items.append(item)
            items = filtered_items

        results: list[CompletedSession] = []
        for item in items:
            session = CompletedSession(
                session_id=item["session_id"],
                first_trace_timestamp_ms=int(item["first_trace_timestamp_ms"]),
                last_trace_timestamp_ms=int(item["last_trace_timestamp_ms"]),
            )
            results.append(session)

        if max_results is not None:
            results = results[:max_results]

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

    @staticmethod
    def _parse_assessment_numeric_value(assess_dict: dict[str, Any]) -> Decimal | None:
        """Parse numeric value from assessment dict for denormalization."""
        fb = assess_dict.get("feedback", {})
        ex = assess_dict.get("expectation", {})
        raw = fb.get("value") if fb else ex.get("value")
        if raw is None:
            return None
        val_str = str(raw)
        if val_str in ("True", "true", "yes"):
            return Decimal("1")
        if val_str in ("False", "false", "no"):
            return Decimal("0")
        try:
            return Decimal(val_str)
        except Exception:
            return None

    def _denormalize_assessment_item(
        self, item: dict[str, Any], assess_dict: dict[str, Any]
    ) -> None:
        """Add denormalized top-level attributes to an assessment item."""
        item["name"] = assess_dict.get("assessment_name", "")
        item["assessment_type"] = "feedback" if "feedback" in assess_dict else "expectation"
        numeric_val = self._parse_assessment_numeric_value(assess_dict)
        if numeric_val is not None:
            item["numeric_value"] = numeric_val
        # Parse created_timestamp from proto timestamp or ISO string
        create_time = assess_dict.get("create_time", {})
        if isinstance(create_time, dict):
            seconds = int(create_time.get("seconds", 0))
            nanos = int(create_time.get("nanos", 0))
            item["created_timestamp"] = seconds * 1000 + nanos // 1_000_000
        elif isinstance(create_time, int | float):
            item["created_timestamp"] = int(create_time)
        elif isinstance(create_time, str) and create_time:
            import datetime as _dt

            try:
                dt = _dt.datetime.fromisoformat(create_time.replace("Z", "+00:00"))
                item["created_timestamp"] = int(dt.timestamp() * 1000)
            except ValueError:
                pass

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
                f"Trace with ID {trace_id} is not found.",
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
        self._denormalize_assessment_item(item, assess_dict)
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
                f"Assessment '{assessment_id}' for trace '{trace_id}' not found.",
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
                f"Assessment '{assessment_id}' for trace '{trace_id}' not found.",
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
        self._denormalize_assessment_item(item, assess_dict)
        self._table.put_item(item)

        # FTS diff
        updated_assessment = Assessment.from_dictionary(assess_dict)
        new_fts_text = self._assessment_fts_text(updated_assessment)
        field = f"assess_{assessment_id}"

        if old_fts_text or new_fts_text:
            levels = ("W", "3", "2")
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
                            forward_sk = f"{SK_FTS_PREFIX}{lvl}#T#{tok}#{trace_id}{field_suffix}"
                            self._table.delete_item(pk=pk, sk=forward_sk)
                            self._table.delete_item(pk=pk, sk=rev_sk)

            # Write new FTS items
            if tokens_to_add:
                new_fts_items: list[dict[str, Any]] = []
                for lvl, tok in tokens_to_add:
                    forward_sk = f"{SK_FTS_PREFIX}{lvl}#T#{tok}#{trace_id}{field_suffix}"
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
                f"Assessment '{assessment_id}' for trace '{trace_id}' not found.",
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
        import json

        from mlflow.genai.scorers.scorer_utils import (
            extract_model_from_serialized_scorer,
            validate_scorer_model,
            validate_scorer_name,
        )

        validate_scorer_name(name)
        serialized_data = json.loads(serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)
        validate_scorer_model(model)

        self.get_experiment(experiment_id)  # verify exists

        # Resolve gateway endpoint references (name → ID)
        import json as _json

        from mlflow.genai.scorers.scorer_utils import (
            build_gateway_model,
            extract_endpoint_ref,
            extract_model_from_serialized_scorer,
            is_gateway_model,
            update_model_in_serialized_scorer,
        )

        serialized_data = _json.loads(serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)
        if is_gateway_model(model):
            assert model is not None  # is_gateway_model returns False for None
            endpoint_name = extract_endpoint_ref(model)
            endpoint = self.get_gateway_endpoint(name=endpoint_name)  # raises if not found
            serialized_data = update_model_in_serialized_scorer(
                serialized_data, build_gateway_model(endpoint.endpoint_id)
            )
            serialized_scorer = _json.dumps(serialized_data)

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
                return self._resolve_endpoint_in_scorer(
                    _ScorerVersionCompat(
                        experiment_id=experiment_id,
                        scorer_name=name,
                        scorer_version=version,
                        serialized_scorer=serialized_scorer,
                        creation_time=now_ms,
                        scorer_id=scorer_id,
                    )
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
        return self._resolve_endpoint_in_scorer(
            _ScorerVersionCompat(
                experiment_id=experiment_id,
                scorer_name=name,
                scorer_version=version,
                serialized_scorer=serialized_scorer,
                creation_time=now_ms,
                scorer_id=scorer_id,
            )
        )

    def get_scorer(
        self, experiment_id: str, name: str, version: int | None = None
    ) -> ScorerVersion:
        scorer_id = self._resolve_scorer_id(experiment_id, name)
        if scorer_id is None:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        meta_sk = f"{SK_SCORER_PREFIX}{scorer_id}"

        if version is not None:
            padded = f"{version:010d}"
            item = self._table.get_item(pk=pk, sk=f"{meta_sk}#V#{padded}")
            if item is None:
                raise MlflowException(
                    f"Scorer with name '{name}' and version {version} not found for "
                    f"experiment {experiment_id}.",
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
        result = _ScorerVersionCompat(
            experiment_id=experiment_id,
            scorer_name=meta["scorer_name"] if meta else name,
            scorer_version=int(item["scorer_version"]),
            serialized_scorer=item["serialized_scorer"],
            creation_time=int(item["creation_time"]),
            scorer_id=scorer_id,
        )
        return self._resolve_endpoint_in_scorer(result)

    def _resolve_endpoint_in_scorer(self, scorer_version: ScorerVersion) -> ScorerVersion:
        """Resolve gateway endpoint ID to name in serialized scorer.

        If the endpoint has been deleted, sets the model field to None.
        """
        import json as _json

        from mlflow.genai.scorers.scorer_utils import (
            build_gateway_model,
            extract_endpoint_ref,
            extract_model_from_serialized_scorer,
            is_gateway_model,
            update_model_in_serialized_scorer,
        )

        serialized_data = _json.loads(scorer_version._serialized_scorer)
        model = extract_model_from_serialized_scorer(serialized_data)
        if not is_gateway_model(model):
            return scorer_version
        assert model is not None  # is_gateway_model returns False for None

        endpoint_ref = extract_endpoint_ref(model)
        try:
            endpoint = self.get_gateway_endpoint(endpoint_id=endpoint_ref)
            new_model: str | None = build_gateway_model(endpoint.name) if endpoint.name else None
        except MlflowException:
            new_model = None

        serialized_data = update_model_in_serialized_scorer(serialized_data, new_model)
        return _ScorerVersionCompat(
            experiment_id=scorer_version.experiment_id,
            scorer_name=scorer_version.scorer_name,
            scorer_version=scorer_version.scorer_version,
            serialized_scorer=_json.dumps(serialized_data),
            creation_time=scorer_version.creation_time,
            scorer_id=scorer_version.scorer_id,
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
                    self._resolve_endpoint_in_scorer(
                        _ScorerVersionCompat(
                            experiment_id=experiment_id,
                            scorer_name=meta["scorer_name"],
                            scorer_version=int(ver["scorer_version"]),
                            serialized_scorer=ver["serialized_scorer"],
                            creation_time=int(ver["creation_time"]),
                            scorer_id=scorer_id,
                        )
                    )
                )
        result.sort(key=lambda sv: sv.scorer_name)
        return result

    def list_scorer_versions(self, experiment_id: str, name: str) -> list[ScorerVersion]:
        scorer_id = self._resolve_scorer_id(experiment_id, name)
        if scorer_id is None:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
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
            self._resolve_endpoint_in_scorer(
                _ScorerVersionCompat(
                    experiment_id=experiment_id,
                    scorer_name=scorer_name,
                    scorer_version=int(item["scorer_version"]),
                    serialized_scorer=item["serialized_scorer"],
                    creation_time=int(item["creation_time"]),
                    scorer_id=scorer_id,
                )
            )
            for item in items
        ]

    def delete_scorer(self, experiment_id: str, name: str, version: int | None = None) -> None:
        scorer_id = self._resolve_scorer_id(experiment_id, name)
        if scorer_id is None:
            raise MlflowException(
                f"Scorer with name '{name}' not found for experiment {experiment_id}.",
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
                    f"Scorer with name '{name}' and version {version} not found for "
                    f"experiment {experiment_id}.",
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

        if not isinstance(sample_rate, int | float):
            raise MlflowException(
                f"sample_rate must be a number, got {type(sample_rate).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not (0.0 <= sample_rate <= 1.0):
            raise MlflowException(
                f"sample_rate must be between 0.0 and 1.0, got {sample_rate}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if filter_string:
            if not isinstance(filter_string, str):
                raise MlflowException(
                    f"filter_string must be a string, got {type(filter_string).__name__}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            from mlflow.utils.search_utils import SearchTraceUtils

            SearchTraceUtils.parse_search_filter_for_search_traces(filter_string)

        scorer_id = self._resolve_scorer_id(experiment_id, scorer_name)
        if scorer_id is None:
            raise MlflowException(
                f"Scorer with name '{scorer_name}' not found for experiment {experiment_id}.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Validate scorer compatibility for online scoring (only when enabling)
        if sample_rate > 0:
            import json

            from mlflow.genai.scorers.scorer_utils import (
                extract_model_from_serialized_scorer,
                is_gateway_model,
            )

            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            # Get latest scorer version
            scorer_meta = self._table.query(
                pk=f"{GSI1_SCOR_PREFIX}{scorer_id}",
                index_name="gsi1",
                limit=1,
            )
            if scorer_meta:
                exp_pk = scorer_meta[0][GSI1_SK]
                scorer_versions = self._table.query(
                    pk=exp_pk,
                    sk_prefix=f"{SK_SCORER_PREFIX}{scorer_id}#V#",
                    scan_forward=False,
                    limit=1,
                )
                if scorer_versions:
                    serialized_data = json.loads(scorer_versions[0].get("serialized_scorer", "{}"))
                    model = extract_model_from_serialized_scorer(serialized_data)
                    if not is_gateway_model(model):
                        raise MlflowException(
                            f"Scorer '{scorer_name}' does not use a gateway model. "
                            "Automatic evaluation is only supported for scorers that use "
                            "gateway models.",
                            INVALID_PARAMETER_VALUE,
                        )

                    # Check if scorer requires expectations
                    try:
                        from mlflow.genai.judges.instructions_judge import (
                            EXPECTATIONS_FIELD,
                            InstructionsJudge,
                        )
                        from mlflow.genai.scorers.base import Scorer

                        scorer_obj = Scorer.model_validate_json(
                            scorer_versions[0].get("serialized_scorer", "{}")
                        )
                        if (
                            isinstance(scorer_obj, InstructionsJudge)
                            and EXPECTATIONS_FIELD in scorer_obj.get_input_fields()
                        ):
                            raise MlflowException(
                                f"Scorer '{scorer_name}' requires expectations, "
                                "but scorers with expectations are not currently "
                                "supported for automatic evaluation.",
                                INVALID_PARAMETER_VALUE,
                            )
                    except Exception:
                        # Fail open on deserialization errors — only re-raise
                        # if it's our own expectations validation error
                        import sys

                        exc = sys.exc_info()[1]
                        if isinstance(exc, MlflowException) and "requires expectations" in str(exc):
                            raise
                        # Otherwise swallow deserialization/import errors

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

            # Filter out scorers whose latest version doesn't use a gateway model
            import json as _json

            from mlflow.genai.scorers.scorer_utils import (
                extract_model_from_serialized_scorer,
                is_gateway_model,
            )

            serialized = versions[0]["serialized_scorer"]
            try:
                model = extract_model_from_serialized_scorer(_json.loads(serialized))
            except Exception:
                model = None
            if not is_gateway_model(model):
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
                    serialized_scorer=serialized,
                    online_config=config,
                )
            )
        return result

    def _delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        """Delete traces and all their sub-items (tags, metadata, assessments, FTS)."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        # Resolve trace_ids from max_timestamp_millis if needed
        if trace_ids is None and max_timestamp_millis is not None:
            from boto3.dynamodb.conditions import Attr

            # Query traces with request_time <= max_timestamp_millis via LSI1
            # LSI1 SK is zero-padded request_time; filter to trace META items only
            items = self._table.query(
                pk=pk,
                index_name="lsi1",
                sk_lte=f"{max_timestamp_millis:020d}",
                sk_gte=f"{0:020d}",
                scan_forward=True,
                filter_expression=Attr("SK").begins_with(SK_TRACE_PREFIX),
            )
            # Only META items (SK = T#<trace_id>, no # after trace_id)
            trace_metas = [item for item in items if "#" not in item["SK"][len(SK_TRACE_PREFIX) :]]
            # Sort oldest first and apply max_traces limit
            trace_metas.sort(key=lambda i: int(i.get("request_time", 0)))
            if max_traces is not None:
                trace_metas = trace_metas[:max_traces]
            trace_ids = [item["SK"][len(SK_TRACE_PREFIX) :] for item in trace_metas]

        if not trace_ids:
            return 0

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
                if parts[0] in ("W", "3", "2"):
                    field_part = ""
                    lvl = parts[0]
                    tok = "#".join(parts[1:])
                else:
                    field_part = f"#{parts[0]}"
                    lvl = parts[1]
                    tok = "#".join(parts[2:])

                forward_sk = f"{SK_FTS_PREFIX}{lvl}#T#{tok}#{trace_id}{field_part}"
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

    @staticmethod
    def _infer_json_type(value: Any) -> str:
        """Infer the JSON type string for a Python value."""
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float | Decimal):
            return "float"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    @staticmethod
    def _compute_dataset_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute schema from record data by inferring types from all records."""
        from mlflow_dynamodbstore.dynamodb.table import convert_decimals

        schema: dict[str, dict[str, str]] = {}
        for record in records:
            for section in ("inputs", "expectations"):
                data = convert_decimals(record.get(section))
                if not isinstance(data, dict):
                    continue
                if section not in schema:
                    schema[section] = {}
                for key, value in data.items():
                    if key not in schema[section]:
                        schema[section][key] = DynamoDBTrackingStore._infer_json_type(value)
        return schema

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
        dataset_id = f"d-{generate_ulid()}"
        digest = self._compute_dataset_digest(name, now_ms)
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"

        # Extract user from mlflow.user tag
        created_by = (tags or {}).get(MLFLOW_USER)

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
            "created_by": created_by or "",
            "last_updated_by": created_by or "",
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
            created_by=created_by,
            last_updated_by=created_by,
        )

    def get_dataset(self, dataset_id: str) -> EvaluationDataset:
        """Fetch an evaluation dataset by ID."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is None:
            raise MlflowException(
                f"Evaluation dataset with id '{dataset_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Load tags from tag items
        tag_items = self._table.query(pk=pk, sk_prefix=SK_DATASET_TAG_PREFIX)
        tags = {item["key"]: item["value"] for item in tag_items}

        return EvaluationDataset(
            dataset_id=dataset_id,
            name=meta["name"],
            digest=meta["digest"],
            created_time=int(meta["created_time"]),
            last_update_time=int(meta["last_update_time"]),
            tags=tags,
            profile=meta.get("profile"),
            schema=meta.get("schema"),
            created_by=meta.get("created_by") or None,
            last_updated_by=meta.get("last_updated_by") or None,
        )

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
                all_datasets.append(_meta_to_dataset(did, meta))
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
                all_datasets.append(_meta_to_dataset(did, meta))

        # --- Apply filter_string in-memory ---
        if filter_string:
            all_datasets = self._apply_dataset_filters(all_datasets, filter_string)

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

    @staticmethod
    def _apply_dataset_filters(
        datasets: list[EvaluationDataset], filter_string: str
    ) -> list[EvaluationDataset]:
        """Apply an in-memory filter to a list of EvaluationDataset objects.

        Supports: name LIKE, name =, created_by =, tags.key =/!=,
        created_time >/</>=/<=/=, AND combinations.
        """
        import re

        # Split on AND (case-insensitive)
        clauses = re.split(r"\s+AND\s+", filter_string.strip(), flags=re.IGNORECASE)

        for clause in clauses:
            clause = clause.strip()

            # name LIKE 'pattern'
            m = re.match(r"""name\s+LIKE\s+['"](.+)['"]\s*$""", clause, re.IGNORECASE)
            if m:
                pattern = m.group(1)
                if pattern.startswith("%") and pattern.endswith("%"):
                    sub = pattern.strip("%")
                    datasets = [d for d in datasets if sub in d.name]
                elif pattern.endswith("%"):
                    pfx = pattern.rstrip("%")
                    datasets = [d for d in datasets if d.name.startswith(pfx)]
                elif pattern.startswith("%"):
                    sfx = pattern.lstrip("%")
                    datasets = [d for d in datasets if d.name.endswith(sfx)]
                else:
                    datasets = [d for d in datasets if d.name == pattern]
                continue

            # name = 'value'
            m = re.match(r"""name\s*=\s*['"](.+)['"]\s*$""", clause, re.IGNORECASE)
            if m:
                val = m.group(1)
                datasets = [d for d in datasets if d.name == val]
                continue

            # created_by = 'value'
            m = re.match(r"""created_by\s*=\s*['"](.+)['"]\s*$""", clause, re.IGNORECASE)
            if m:
                val = m.group(1)
                datasets = [d for d in datasets if d.created_by == val]
                continue

            # last_updated_by = 'value'
            m = re.match(r"""last_updated_by\s*=\s*['"](.+)['"]\s*$""", clause, re.IGNORECASE)
            if m:
                val = m.group(1)
                datasets = [d for d in datasets if d.last_updated_by == val]
                continue

            # tags.key = 'value' or tags.key != 'value'
            m = re.match(r"""tags\.(\S+)\s*(!=|=)\s*['"](.+)['"]\s*$""", clause, re.IGNORECASE)
            if m:
                tag_key, op, tag_val = m.group(1), m.group(2), m.group(3)
                if op == "=":
                    datasets = [d for d in datasets if (d.tags or {}).get(tag_key) == tag_val]
                else:
                    datasets = [d for d in datasets if (d.tags or {}).get(tag_key) != tag_val]
                continue

            # created_time >/</>=/<=/= N
            m = re.match(r"""created_time\s*(>=|<=|>|<|=)\s*(\d+)\s*$""", clause, re.IGNORECASE)
            if m:
                op, val = m.group(1), int(m.group(2))
                if op == ">":
                    datasets = [d for d in datasets if d.created_time > val]
                elif op == ">=":
                    datasets = [d for d in datasets if d.created_time >= val]
                elif op == "<":
                    datasets = [d for d in datasets if d.created_time < val]
                elif op == "<=":
                    datasets = [d for d in datasets if d.created_time <= val]
                elif op == "=":
                    datasets = [d for d in datasets if d.created_time == val]
                continue

            # Unrecognized filter clause
            attr = clause.split()[0] if clause.split() else clause
            raise MlflowException(
                f"Invalid attribute key '{attr}' specified.",
                INVALID_PARAMETER_VALUE,
            )

        return datasets

    def set_dataset_tags(self, dataset_id: str, tags: dict[str, str]) -> None:
        """Set (upsert) one or more tags on an evaluation dataset.

        Tags with None value are ignored (not written, not deleted).
        Tag updates do NOT change last_update_time or last_updated_by.
        """
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        for key, value in tags.items():
            if value is None:
                continue
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
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is None:
            raise MlflowException(
                f"Dataset '{dataset_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        # Validate experiments exist
        for exp_id in experiment_ids:
            self.get_experiment(exp_id)  # raises if not found
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
        return self.get_dataset(dataset_id)

    def remove_dataset_from_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> EvaluationDataset:
        """Remove an evaluation dataset's association from one or more experiments."""
        pk = f"{PK_DATASET_PREFIX}{dataset_id}"
        meta = self._table.get_item(pk=pk, sk=SK_DATASET_META)
        if meta is None:
            raise MlflowException(
                f"Dataset '{dataset_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        for exp_id in experiment_ids:
            sk = f"{SK_DATASET_EXP_PREFIX}{exp_id}"
            existing = self._table.get_item(pk=pk, sk=sk)
            if existing is None:
                _logger.warning(
                    "Dataset '%s' was not associated with experiment '%s'.",
                    dataset_id,
                    exp_id,
                )
                continue
            self._table.delete_item(pk=pk, sk=sk)
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
                f"Evaluation dataset with id '{dataset_id}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        inserted = 0
        updated = 0

        from mlflow_dynamodbstore.dynamodb.table import convert_floats

        for record in records:
            raw_inputs = record.get("inputs", {})
            # Compute input_hash before float→Decimal conversion
            input_hash = hashlib.sha256(
                _json.dumps(raw_inputs, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:8]

            inputs = convert_floats(raw_inputs)
            outputs = convert_floats(record.get("outputs"))
            expectations = convert_floats(record.get("expectations"))
            record_tags = convert_floats(record.get("tags"))
            source = record.get("source")

            # Dedup: query LSI3 for existing record with same input_hash
            existing_items, _ = self._table.query_page(
                pk=pk,
                sk_prefix=input_hash,
                index_name="lsi3",
                filter_expression=Attr("SK").begins_with(SK_DATASET_RECORD_PREFIX),
            )

            now_ms = int(time.time() * 1000)

            if existing_items:
                # Update the first matching record (merge expectations and tags)
                existing = existing_items[0]
                updates: dict[str, Any] = {
                    "last_update_time": now_ms,
                    LSI2_SK: now_ms,
                }
                if outputs is not None:
                    updates["outputs"] = outputs
                if expectations is not None:
                    merged_exp = dict(existing.get("expectations") or {})
                    merged_exp.update(expectations)
                    updates["expectations"] = merged_exp
                if record_tags is not None:
                    merged_tags = dict(existing.get("tags") or {})
                    merged_tags.update(record_tags)
                    updates["tags"] = merged_tags
                # Preserve source from first write
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

        # Compute schema from all records
        schema = self._compute_dataset_schema(all_records)

        now_ms = int(time.time() * 1000)
        digest = self._compute_dataset_digest(meta["name"], now_ms)
        meta_updates: dict[str, Any] = {
            "profile": _json.dumps({"num_records": num_records}),
            "schema": _json.dumps(schema),
            "last_update_time": now_ms,
            "digest": digest,
            LSI2_SK: now_ms,
        }
        # Update last_updated_by if any record has mlflow.user tag
        for record in records:
            user = (record.get("tags") or {}).get(MLFLOW_USER)
            if user:
                meta_updates["last_updated_by"] = user
                break
        self._table.update_item(pk=pk, sk=SK_DATASET_META, updates=meta_updates)

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

        from mlflow.entities.dataset_record_source import DatasetRecordSource

        from mlflow_dynamodbstore.dynamodb.table import convert_decimals

        record_list = []
        for item in items:
            raw_source = convert_decimals(item.get("source"))
            source = None
            if isinstance(raw_source, dict) and "source_type" in raw_source:
                source = DatasetRecordSource(
                    source_type=raw_source["source_type"],
                    source_data=raw_source.get("source_data"),
                )
            record_list.append(
                DatasetRecord(
                    dataset_id=dataset_id,
                    dataset_record_id=item["dataset_record_id"],
                    inputs=convert_decimals(item.get("inputs", {})),
                    created_time=int(item["created_time"]),
                    last_update_time=int(item["last_update_time"]),
                    outputs=convert_decimals(item.get("outputs")),
                    expectations=convert_decimals(item.get("expectations")),
                    tags=convert_decimals(item.get("tags")),
                    source=source,
                )
            )

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
        # Check which records actually exist before deleting
        existing_keys = []
        for rec_id in record_ids:
            sk = f"{SK_DATASET_RECORD_PREFIX}{rec_id}"
            if self._table.get_item(pk=pk, sk=sk) is not None:
                existing_keys.append({"PK": pk, "SK": sk})
        if existing_keys:
            self._table.batch_delete(existing_keys)

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

        return len(existing_keys)

    # ------------------------------------------------------------------
    # Trace Metrics Query
    # ------------------------------------------------------------------

    def query_trace_metrics(
        self,
        experiment_ids: list[str],
        view_type: MetricViewType,
        metric_name: str,
        aggregations: list[MetricAggregation],
        dimensions: list[str] | None = None,
        filters: list[str] | None = None,
        time_interval_seconds: int | None = None,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        max_results: int = 1000,
        page_token: str | None = None,
    ) -> PagedList[list[MetricDataPoint]]:
        """Query aggregated trace metrics across experiments."""
        from mlflow.entities.trace_metrics import AggregationType, MetricViewType
        from mlflow.store.tracking.utils.sql_trace_metrics_utils import (
            validate_query_trace_metrics_params,
        )

        from mlflow_dynamodbstore.trace_metrics.accumulators import MetricAccumulator
        from mlflow_dynamodbstore.trace_metrics.extractors import (
            TIME_BUCKET_LABEL,
            build_dimension_key,
            compute_time_bucket,
            extract_metric_value,
            get_timestamp_for_view,
        )
        from mlflow_dynamodbstore.trace_metrics.filters import (
            apply_trace_metric_filters,
            filter_assessment_items,
            filter_span_items,
            meta_prefilter_spans,
        )
        from mlflow_dynamodbstore.trace_metrics.pagination import (
            cache_get,
            cache_put,
            compute_query_hash,
            decode_page_token,
            encode_page_token,
        )

        # 1. VALIDATE
        validate_query_trace_metrics_params(view_type, metric_name, aggregations, dimensions)
        if time_interval_seconds is not None and (start_time_ms is None or end_time_ms is None):
            raise MlflowException(
                "start_time_ms and end_time_ms are required when time_interval_seconds is set.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # 2. CHECK CACHE (if page_token)
        if page_token:
            token_data = decode_page_token(page_token)
            cached = cache_get(self._table, token_data["query_hash"])
            if cached is not None:
                offset = token_data["offset"]
                page = cached[offset : offset + max_results]
                next_token = None
                if offset + max_results < len(cached):
                    next_token = encode_page_token(token_data["query_hash"], offset + max_results)
                return PagedList(page, next_token)  # type: ignore[arg-type]

        # Need percentile values?
        need_values = any(a.aggregation_type == AggregationType.PERCENTILE for a in aggregations)

        # 3. STREAM TRACE META ITEMS and accumulate
        accumulators: dict[tuple[str | None, ...], MetricAccumulator] = {}
        dim_labels: dict[tuple[str | None, ...], dict[str, str]] = {}

        for experiment_id in experiment_ids:
            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

            # Query META items (with optional time range via LSI1)
            if start_time_ms is not None and end_time_ms is not None:
                meta_candidates = self._table.query(
                    pk=pk,
                    index_name="lsi1",
                    sk_gte=f"{start_time_ms:020d}",
                    sk_lte=f"{end_time_ms:020d}",
                )
            else:
                # NOTE: Without time range, this fetches all T# items (including sub-items).
                # The "request_time" in item filter discards non-META items in Python.
                # This is known technical debt — a dedicated META-only
                # index would be more efficient.
                meta_candidates = self._table.query(pk=pk, sk_prefix=SK_TRACE_PREFIX)

            # Filter to META items only (have "request_time" attribute)
            meta_items = [item for item in meta_candidates if "request_time" in item]

            for meta_item in meta_items:
                trace_id = meta_item["SK"][len(SK_TRACE_PREFIX) :]

                # Only query tags if needed for filters or trace_name dimension
                needs_tags = False
                if filters:
                    needs_tags = any("tag" in f for f in filters)
                if dimensions and "trace_name" in dimensions:
                    needs_tags = True
                needs_metadata = bool(filters and any("metadata" in f for f in filters))

                tag_items = (
                    self._table.query(pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#TAG#")
                    if needs_tags
                    else []
                )
                metadata_items = (
                    self._table.query(pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#RMETA#")
                    if needs_metadata
                    else []
                )

                # Apply trace-level filters
                if not apply_trace_metric_filters(
                    meta_item, filters, view_type, tag_items, metadata_items
                ):
                    continue

                # Build trace_tags dict for dimension extraction
                trace_tags = {t["key"]: t["value"] for t in tag_items}

                if view_type == MetricViewType.TRACES:
                    # Fetch TMETRIC items for token metrics
                    tmetric_items = self._table.query(
                        pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}{SK_TRACE_METRIC_PREFIX}"
                    )

                    # Compute time bucket
                    time_bucket = None
                    if time_interval_seconds is not None:
                        ts = get_timestamp_for_view(view_type, meta_item, meta_item)
                        time_bucket = compute_time_bucket(ts, time_interval_seconds)

                    value = extract_metric_value(
                        metric_name, view_type, meta_item, meta_item, tmetric_items
                    )
                    if value is None:
                        continue

                    dim_key = build_dimension_key(
                        dimensions, view_type, meta_item, meta_item, trace_tags, time_bucket
                    )
                    # Skip if any dimension value is None
                    if dimensions and any(v is None for v in dim_key):
                        continue

                    if dim_key not in accumulators:
                        accumulators[dim_key] = MetricAccumulator(collect_values=need_values)
                        # Build dimension labels
                        labels: dict[str, str] = {}
                        idx = 0
                        if time_bucket is not None:
                            labels[TIME_BUCKET_LABEL] = time_bucket
                            idx = 1
                        for d in dimensions or []:
                            labels[d] = str(dim_key[idx]) if dim_key[idx] is not None else ""
                            idx += 1
                        dim_labels[dim_key] = labels
                    accumulators[dim_key].add(value)

                elif view_type == MetricViewType.SPANS:
                    # Pre-filter using META denormalized fields
                    if not meta_prefilter_spans(meta_item, filters):
                        continue

                    # Fetch span items and span metric items
                    span_items = self._table.query(
                        pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}{SK_SPAN_PREFIX}"
                    )
                    smetric_items = self._table.query(
                        pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}{SK_SPAN_METRIC_PREFIX}"
                    )

                    # Apply span-level filters
                    span_items = filter_span_items(span_items, filters)

                    for span_item in span_items:
                        time_bucket = None
                        if time_interval_seconds is not None:
                            ts = get_timestamp_for_view(view_type, span_item, meta_item)
                            time_bucket = compute_time_bucket(ts, time_interval_seconds)

                        value = extract_metric_value(
                            metric_name,
                            view_type,
                            span_item,
                            meta_item,
                            span_metric_items=smetric_items,
                        )
                        if value is None:
                            continue

                        dim_key = build_dimension_key(
                            dimensions, view_type, span_item, meta_item, trace_tags, time_bucket
                        )
                        if dimensions and any(v is None for v in dim_key):
                            continue

                        if dim_key not in accumulators:
                            accumulators[dim_key] = MetricAccumulator(collect_values=need_values)
                            labels = {}
                            idx = 0
                            if time_bucket is not None:
                                labels[TIME_BUCKET_LABEL] = time_bucket
                                idx = 1
                            for d in dimensions or []:
                                labels[d] = str(dim_key[idx]) if dim_key[idx] is not None else ""
                                idx += 1
                            dim_labels[dim_key] = labels
                        accumulators[dim_key].add(value)

                elif view_type == MetricViewType.ASSESSMENTS:
                    # Fetch assessment items
                    assess_items = self._table.query(
                        pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#"
                    )

                    # Apply assessment-level filters
                    assess_items = filter_assessment_items(assess_items, filters)

                    for assess_item in assess_items:
                        time_bucket = None
                        if time_interval_seconds is not None:
                            ts = get_timestamp_for_view(view_type, assess_item, meta_item)
                            time_bucket = compute_time_bucket(ts, time_interval_seconds)

                        value = extract_metric_value(metric_name, view_type, assess_item, meta_item)
                        if value is None:
                            continue

                        dim_key = build_dimension_key(
                            dimensions, view_type, assess_item, meta_item, trace_tags, time_bucket
                        )
                        if dimensions and any(v is None for v in dim_key):
                            continue

                        if dim_key not in accumulators:
                            accumulators[dim_key] = MetricAccumulator(collect_values=need_values)
                            labels = {}
                            idx = 0
                            if time_bucket is not None:
                                labels[TIME_BUCKET_LABEL] = time_bucket
                                idx = 1
                            for d in dimensions or []:
                                labels[d] = str(dim_key[idx]) if dim_key[idx] is not None else ""
                                idx += 1
                            dim_labels[dim_key] = labels
                        accumulators[dim_key].add(value)

        # 4. FINALIZE — compute aggregation results, convert to MetricDataPoint list
        from mlflow.entities.trace_metrics import MetricDataPoint

        all_data_points: list[MetricDataPoint] = []
        for dim_key, acc in accumulators.items():
            agg_values = acc.finalize(aggregations)
            labels = dim_labels.get(dim_key, {})
            dp = MetricDataPoint(
                metric_name=metric_name,
                dimensions=labels or {},
                values=agg_values,
            )
            all_data_points.append(dp)

        # 5. CACHE AND PAGINATE
        query_hash = compute_query_hash(
            experiment_ids,
            view_type,
            metric_name,
            aggregations,
            dimensions,
            filters,
            time_interval_seconds,
            start_time_ms,
            end_time_ms,
        )
        offset = 0
        if page_token:
            from mlflow_dynamodbstore.trace_metrics.pagination import decode_page_token as _decode

            token_data = _decode(page_token)
            offset = token_data.get("offset", 0)

        cache_put(self._table, query_hash, all_data_points)
        page = all_data_points[offset : offset + max_results]
        next_offset = offset + max_results
        next_token = (
            encode_page_token(query_hash, next_offset)
            if next_offset < len(all_data_points)
            else None
        )
        return PagedList(page, next_token)  # type: ignore[arg-type]

    # -----------------------------------------------------------------------
    # Gateway Secrets
    # -----------------------------------------------------------------------

    def _invalidate_secret_cache(self) -> None:
        """Invalidate the gateway secret cache. No-op until cache is implemented."""
        pass

    def create_gateway_secret(
        self,
        secret_name: str,
        secret_value: dict[str, str],
        provider: str | None = None,
        auth_config: dict[str, str] | None = None,
        created_by: str | None = None,
    ) -> GatewaySecretInfo:
        import json as _json

        # Check name uniqueness via GSI1
        existing = self._table.query(
            pk=f"{GSI1_GW_SECRET_NAME_PREFIX}{self._workspace}#{secret_name}",
            index_name="gsi1",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Secret with name '{secret_name}' already exists",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        secret_id = f"s-{generate_ulid()}"
        now = get_current_time_millis()

        # Encrypt
        kek_manager = KEKManager()
        value_to_encrypt = _json.dumps(secret_value)
        encrypted = _encrypt_secret(value_to_encrypt, kek_manager, secret_id, secret_name)
        masked_value = _mask_secret_value(secret_value)

        item = {
            "PK": f"{PK_GW_SECRET_PREFIX}{secret_id}",
            "SK": SK_GW_META,
            "secret_name": secret_name,
            "encrypted_value": encrypted.encrypted_value,
            "wrapped_dek": encrypted.wrapped_dek,
            "kek_version": encrypted.kek_version,
            "masked_value": masked_value,
            "created_at": now,
            "last_updated_at": now,
            "workspace": self._workspace,
            # GSI projections
            "gsi1pk": f"{GSI1_GW_SECRET_NAME_PREFIX}{self._workspace}#{secret_name}",
            "gsi1sk": secret_id,
            "gsi2pk": f"{GSI2_GW_SECRETS_PREFIX}{self._workspace}",
            "gsi2sk": secret_id,
        }
        if provider is not None:
            item["provider"] = provider
        if auth_config is not None:
            item["auth_config"] = auth_config
        if created_by is not None:
            item["created_by"] = created_by
            item["last_updated_by"] = created_by

        self._table.put_item(item)
        self._invalidate_secret_cache()

        return GatewaySecretInfo(
            secret_id=secret_id,
            secret_name=secret_name,
            masked_values=masked_value,
            created_at=now,
            last_updated_at=now,
            provider=provider,
            auth_config=auth_config,
            workspace=self._workspace,
            created_by=created_by,
            last_updated_by=created_by,
        )

    def _get_secret_item(self, secret_id: str) -> dict[str, Any] | None:
        """Fetch raw DynamoDB item for a secret by ID. Returns None if not found."""
        return self._table.get_item(
            pk=f"{PK_GW_SECRET_PREFIX}{secret_id}",
            sk=SK_GW_META,
        )

    def _secret_item_to_entity(self, item: dict[str, Any]) -> GatewaySecretInfo:
        """Convert a raw DynamoDB secret item to a GatewaySecretInfo entity."""
        secret_id = item["PK"].removeprefix(PK_GW_SECRET_PREFIX)
        return GatewaySecretInfo(
            secret_id=secret_id,
            secret_name=item["secret_name"],
            masked_values=item["masked_value"],
            created_at=int(item["created_at"]),
            last_updated_at=int(item["last_updated_at"]),
            provider=item.get("provider"),
            auth_config=item.get("auth_config"),
            workspace=item.get("workspace"),
            created_by=item.get("created_by"),
            last_updated_by=item.get("last_updated_by"),
        )

    def get_secret_info(
        self,
        secret_id: str | None = None,
        secret_name: str | None = None,
    ) -> GatewaySecretInfo:
        # Validate exactly one of secret_id or secret_name
        if (secret_id is None) == (secret_name is None):
            raise MlflowException(
                "Exactly one of `secret_id` or `secret_name` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if secret_id:
            item = self._get_secret_item(secret_id)
        else:
            # Look up by name via GSI1
            results = self._table.query(
                pk=f"{GSI1_GW_SECRET_NAME_PREFIX}{self._workspace}#{secret_name}",
                index_name="gsi1",
                limit=1,
            )
            if not results:
                raise MlflowException(
                    f"Secret with name '{secret_name}' not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            found_secret_id = results[0]["gsi1sk"]
            item = self._get_secret_item(found_secret_id)

        if item is None:
            identifier = secret_id if secret_id else secret_name
            raise MlflowException(
                f"Secret '{identifier}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return self._secret_item_to_entity(item)

    def update_gateway_secret(
        self,
        secret_id: str,
        secret_value: dict[str, str] | None = None,
        auth_config: dict[str, str] | None = None,
        updated_by: str | None = None,
    ) -> GatewaySecretInfo:
        import json as _json

        # Fetch existing to verify it exists and get secret_name for AAD
        item = self._get_secret_item(secret_id)
        if item is None:
            raise MlflowException(
                f"Secret '{secret_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()
        updates: dict[str, Any] = {"last_updated_at": now}
        removes: list[str] = []

        if secret_value is not None:
            kek_manager = KEKManager()
            value_to_encrypt = _json.dumps(secret_value)
            encrypted = _encrypt_secret(
                value_to_encrypt, kek_manager, secret_id, item["secret_name"]
            )
            updates["encrypted_value"] = encrypted.encrypted_value
            updates["wrapped_dek"] = encrypted.wrapped_dek
            updates["kek_version"] = encrypted.kek_version
            updates["masked_value"] = _mask_secret_value(secret_value)

        if auth_config is not None:
            if auth_config:
                updates["auth_config"] = auth_config
            else:
                # Empty dict means clear auth_config
                removes.append("auth_config")

        if updated_by is not None:
            updates["last_updated_by"] = updated_by

        self._table.update_item(
            pk=f"{PK_GW_SECRET_PREFIX}{secret_id}",
            sk=SK_GW_META,
            updates=updates,
            removes=removes if removes else None,
        )

        # Re-fetch to return the full updated entity
        updated_item = self._get_secret_item(secret_id)
        self._invalidate_secret_cache()
        return self._secret_item_to_entity(updated_item)  # type: ignore[arg-type]

    def delete_gateway_secret(self, secret_id: str) -> None:
        item = self._get_secret_item(secret_id)
        if item is None:
            raise MlflowException(
                f"Secret '{secret_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Orphan model definitions that reference this secret (SET NULL behavior)
        model_defs = self._table.query(
            pk=f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}",
            index_name="gsi4",
        )
        for md_item in model_defs:
            self._table.update_item(
                pk=md_item["PK"],
                sk=md_item["SK"],
                removes=["secret_id", "gsi4pk", "gsi4sk"],
            )

        self._table.delete_item(
            pk=f"{PK_GW_SECRET_PREFIX}{secret_id}",
            sk=SK_GW_META,
        )
        self._invalidate_secret_cache()

    def list_secret_infos(self, provider: str | None = None) -> list[GatewaySecretInfo]:
        from boto3.dynamodb.conditions import Attr

        filter_expr = Attr("provider").eq(provider) if provider else None

        items = self._table.query(
            pk=f"{GSI2_GW_SECRETS_PREFIX}{self._workspace}",
            index_name="gsi2",
            filter_expression=filter_expr,
        )

        # GSI2 has ALL projection — items include all base table attributes
        # (PK, SK, secret_name, masked_value, etc.), no re-fetch needed.
        return [self._secret_item_to_entity(item) for item in items]

    # -----------------------------------------------------------------------
    # Gateway Model Definitions
    # -----------------------------------------------------------------------

    def _resolve_secret_name(self, secret_id: str | None) -> str | None:
        """Resolve secret_name from secret_id. Returns None if secret_id is None or not found."""
        if not secret_id:
            return None
        item = self._get_secret_item(secret_id)
        return item["secret_name"] if item else None

    def create_gateway_model_definition(
        self,
        name: str,
        secret_id: str,
        provider: str,
        model_name: str,
        created_by: str | None = None,
    ) -> GatewayModelDefinition:
        # Check name uniqueness via GSI3
        existing = self._table.query(
            pk=f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Model definition with name '{name}' already exists",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        # Verify secret exists
        secret_item = self._get_secret_item(secret_id)
        if secret_item is None:
            raise MlflowException(
                f"Secret '{secret_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        secret_name = secret_item["secret_name"]

        model_definition_id = f"d-{generate_ulid()}"
        now = get_current_time_millis()

        item = {
            "PK": f"{PK_GW_MODELDEF_PREFIX}{model_definition_id}",
            "SK": SK_GW_META,
            "name": name,
            "secret_id": secret_id,
            "provider": provider,
            "model_name": model_name,
            "created_at": now,
            "last_updated_at": now,
            "workspace": self._workspace,
            # GSI projections
            "gsi2pk": f"{GSI2_GW_MODELDEFS_PREFIX}{self._workspace}",
            "gsi2sk": model_definition_id,
            "gsi3pk": f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}",
            "gsi3sk": model_definition_id,
            "gsi4pk": f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}",
            "gsi4sk": model_definition_id,
        }
        if created_by is not None:
            item["created_by"] = created_by
            item["last_updated_by"] = created_by

        self._table.put_item(item)

        return GatewayModelDefinition(
            model_definition_id=model_definition_id,
            name=name,
            secret_id=secret_id,
            secret_name=secret_name,
            provider=provider,
            model_name=model_name,
            created_at=now,
            last_updated_at=now,
            created_by=created_by,
            last_updated_by=created_by,
            workspace=self._workspace,
        )

    def _get_model_def_item(self, model_definition_id: str) -> dict[str, Any] | None:
        """Fetch raw DynamoDB item for a model definition by ID. Returns None if not found."""
        return self._table.get_item(
            pk=f"{PK_GW_MODELDEF_PREFIX}{model_definition_id}",
            sk=SK_GW_META,
        )

    def _model_def_item_to_entity(
        self, item: dict[str, Any], *, secret_name: Any = _UNSET
    ) -> GatewayModelDefinition:
        """Convert a raw DynamoDB model definition item to a GatewayModelDefinition entity.

        Args:
            item: Raw DynamoDB item.
            secret_name: Pre-resolved secret name. If not provided, resolved via GetItem.
        """
        model_definition_id = item["PK"].removeprefix(PK_GW_MODELDEF_PREFIX)
        secret_id = item.get("secret_id")
        if secret_name is _UNSET:
            secret_name = self._resolve_secret_name(secret_id)
        return GatewayModelDefinition(
            model_definition_id=model_definition_id,
            name=item["name"],
            secret_id=secret_id,
            secret_name=secret_name,
            provider=item["provider"],
            model_name=item["model_name"],
            created_at=int(item["created_at"]),
            last_updated_at=int(item["last_updated_at"]),
            created_by=item.get("created_by"),
            last_updated_by=item.get("last_updated_by"),
            workspace=item.get("workspace"),
        )

    def get_gateway_model_definition(
        self,
        model_definition_id: str | None = None,
        name: str | None = None,
    ) -> GatewayModelDefinition:
        if (model_definition_id is None) == (name is None):
            raise MlflowException(
                "Exactly one of `model_definition_id` or `name` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if model_definition_id:
            item = self._get_model_def_item(model_definition_id)
        else:
            results = self._table.query(
                pk=f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}",
                index_name="gsi3",
                limit=1,
            )
            if not results:
                raise MlflowException(
                    f"Model definition with name '{name}' not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            found_id = results[0]["gsi3sk"]
            item = self._get_model_def_item(found_id)

        if item is None:
            identifier = model_definition_id if model_definition_id else name
            raise MlflowException(
                f"Model definition '{identifier}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return self._model_def_item_to_entity(item)

    def list_gateway_model_definitions(
        self,
        provider: str | None = None,
        secret_id: str | None = None,
    ) -> list[GatewayModelDefinition]:
        if secret_id:
            # Direct lookup via GSI4
            items = self._table.query(
                pk=f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}",
                index_name="gsi4",
            )
        else:
            # List all via GSI2
            items = self._table.query(
                pk=f"{GSI2_GW_MODELDEFS_PREFIX}{self._workspace}",
                index_name="gsi2",
            )

        # GSI has ALL projection — items include all base table attributes.
        # Batch-resolve secret_names to avoid N+1 GetItem calls.
        unique_secret_ids = {item.get("secret_id") for item in items if item.get("secret_id")}
        secret_names = {sid: self._resolve_secret_name(sid) for sid in unique_secret_ids}
        model_defs = [
            self._model_def_item_to_entity(
                item, secret_name=secret_names.get(item.get("secret_id"))
            )
            for item in items
        ]

        # Apply provider filter in-memory if specified
        if provider is not None:
            model_defs = [md for md in model_defs if md.provider == provider]

        return model_defs

    def update_gateway_model_definition(
        self,
        model_definition_id: str,
        name: str | None = None,
        secret_id: Any = _UNSET,
        model_name: str | None = None,
        updated_by: str | None = None,
        provider: str | None = None,
    ) -> GatewayModelDefinition:
        item = self._get_model_def_item(model_definition_id)
        if item is None:
            raise MlflowException(
                f"Model definition '{model_definition_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()
        updates: dict[str, Any] = {"last_updated_at": now}
        removes: list[str] = []

        if name is not None:
            # Check new name uniqueness
            existing = self._table.query(
                pk=f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}",
                index_name="gsi3",
                limit=1,
            )
            if existing and existing[0]["gsi3sk"] != model_definition_id:
                raise MlflowException(
                    f"Model definition with name '{name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
            updates["name"] = name
            updates["gsi3pk"] = f"{GSI3_GW_MODELDEF_NAME_PREFIX}{self._workspace}#{name}"

        if secret_id is not _UNSET:
            if secret_id is None:
                # Intentional orphan — remove secret_id and GSI4 projection
                removes.extend(["secret_id", "gsi4pk", "gsi4sk"])
            else:
                # Verify new secret exists
                secret_item = self._get_secret_item(secret_id)
                if secret_item is None:
                    raise MlflowException(
                        f"Secret '{secret_id}' not found",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    )
                updates["secret_id"] = secret_id
                updates["gsi4pk"] = f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}"
                updates["gsi4sk"] = model_definition_id

        if model_name is not None:
            updates["model_name"] = model_name

        if provider is not None:
            updates["provider"] = provider

        if updated_by is not None:
            updates["last_updated_by"] = updated_by

        self._table.update_item(
            pk=f"{PK_GW_MODELDEF_PREFIX}{model_definition_id}",
            sk=SK_GW_META,
            updates=updates,
            removes=removes if removes else None,
        )

        updated_item = self._get_model_def_item(model_definition_id)
        assert updated_item is not None  # just wrote it
        self._invalidate_secret_cache()
        return self._model_def_item_to_entity(updated_item)

    def delete_gateway_model_definition(self, model_definition_id: str) -> None:
        item = self._get_model_def_item(model_definition_id)
        if item is None:
            raise MlflowException(
                f"Model definition '{model_definition_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # RESTRICT: check if any endpoints use this model def via GSI5
        endpoints = self._table.query(
            pk=f"{GSI5_GW_ENDPOINT_MODELDEF_PREFIX}{model_definition_id}",
            index_name="gsi5",
        )
        if endpoints:
            raise MlflowException(
                "Cannot delete model definition that is currently in use by endpoints. "
                "Detach it from all endpoints first.",
                error_code=INVALID_STATE,
            )

        self._table.delete_item(
            pk=f"{PK_GW_MODELDEF_PREFIX}{model_definition_id}",
            sk=SK_GW_META,
        )
        self._invalidate_secret_cache()

    # -----------------------------------------------------------------------
    # Gateway Endpoints
    # -----------------------------------------------------------------------

    def _get_endpoint_items(self, endpoint_id: str) -> list[dict[str, Any]]:
        """Query all items in an endpoint partition (META + mappings + bindings + tags)."""
        return self._table.query(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
        )

    def _endpoint_items_to_entity(
        self, endpoint_id: str, items: list[dict[str, Any]]
    ) -> GatewayEndpoint:
        """Convert a list of raw DynamoDB items from an endpoint partition to a GatewayEndpoint."""
        meta: dict[str, Any] | None = None
        mapping_items: list[dict[str, Any]] = []
        tag_items: list[dict[str, Any]] = []

        for item in items:
            sk = item["SK"]
            if sk == SK_GW_META:
                meta = item
            elif sk.startswith(SK_GW_MAP_PREFIX):
                mapping_items.append(item)
            elif sk.startswith(SK_GW_TAG_PREFIX):
                tag_items.append(item)

        if meta is None:
            raise MlflowException(
                f"Endpoint '{endpoint_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Resolve model definitions for mappings (deduplicate to avoid redundant fetches)
        model_def_ids = {m["model_definition_id"] for m in mapping_items}
        model_def_cache: dict[str, GatewayModelDefinition] = {}
        for md_id in model_def_ids:
            md_item = self._get_model_def_item(md_id)
            if md_item is not None:
                model_def_cache[md_id] = self._model_def_item_to_entity(md_item)

        model_mappings = []
        for m_item in mapping_items:
            md_id = m_item["model_definition_id"]
            model_mappings.append(
                GatewayEndpointModelMapping(
                    mapping_id=m_item["mapping_id"],
                    endpoint_id=endpoint_id,
                    model_definition_id=md_id,
                    model_definition=model_def_cache.get(md_id),
                    weight=float(m_item["weight"]),
                    linkage_type=GatewayModelLinkageType(m_item["linkage_type"]),
                    fallback_order=int(m_item["fallback_order"])
                    if m_item.get("fallback_order") is not None
                    else None,
                    created_at=int(m_item["created_at"]),
                    created_by=m_item.get("created_by"),
                )
            )

        tags = [
            GatewayEndpointTag(
                key=t["key"],
                value=t.get("value"),
            )
            for t in tag_items
        ]

        # Reconstruct routing_strategy
        routing_strategy = None
        if meta.get("routing_strategy"):
            routing_strategy = RoutingStrategy(meta["routing_strategy"])

        # Reconstruct fallback_config
        fallback_config = None
        if meta.get("fallback_config"):
            fc = meta["fallback_config"]
            fallback_config = FallbackConfig(
                strategy=FallbackStrategy(fc["strategy"]) if fc.get("strategy") else None,
                max_attempts=int(fc["max_attempts"])
                if fc.get("max_attempts") is not None
                else None,
            )

        return GatewayEndpoint(
            endpoint_id=endpoint_id,
            name=meta.get("name"),
            created_at=int(meta["created_at"]),
            last_updated_at=int(meta["last_updated_at"]),
            model_mappings=model_mappings,
            tags=tags,
            created_by=meta.get("created_by"),
            last_updated_by=meta.get("last_updated_by"),
            routing_strategy=routing_strategy,
            fallback_config=fallback_config,
            experiment_id=str(meta["experiment_id"])
            if meta.get("experiment_id") is not None
            else None,
            usage_tracking=bool(meta.get("usage_tracking", False)),
            workspace=meta.get("workspace"),
        )

    def _build_mapping_item(
        self,
        endpoint_id: str,
        config: GatewayEndpointModelConfig,
        mapping_id: str,
        now: int,
        created_by: str | None,
    ) -> dict[str, Any]:
        """Build a DynamoDB mapping item from a GatewayEndpointModelConfig."""
        item: dict[str, Any] = {
            "PK": f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            "SK": f"{SK_GW_MAP_PREFIX}{config.model_definition_id}#{config.linkage_type.value}",
            "model_definition_id": config.model_definition_id,
            "weight": Decimal(str(config.weight)),
            "linkage_type": config.linkage_type.value,
            "created_at": now,
            "mapping_id": mapping_id,
            # GSI5 projection for RESTRICT delete checks
            "gsi5pk": f"{GSI5_GW_ENDPOINT_MODELDEF_PREFIX}{config.model_definition_id}",
            "gsi5sk": endpoint_id,
        }
        if config.fallback_order is not None:
            item["fallback_order"] = config.fallback_order
        if created_by is not None:
            item["created_by"] = created_by
        return item

    def create_gateway_endpoint(
        self,
        name: str,
        model_configs: list[GatewayEndpointModelConfig],
        created_by: str | None = None,
        routing_strategy: RoutingStrategy | None = None,
        fallback_config: FallbackConfig | None = None,
        experiment_id: str | None = None,
        usage_tracking: bool = False,
    ) -> GatewayEndpoint:
        if not model_configs:
            raise MlflowException(
                "Endpoint must have at least one model configuration",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Check name uniqueness via GSI3
        existing = self._table.query(
            pk=f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Endpoint with name '{name}' already exists",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        # Validate all model definitions exist
        for config in model_configs:
            md_item = self._get_model_def_item(config.model_definition_id)
            if md_item is None:
                raise MlflowException(
                    f"Model definitions not found: {config.model_definition_id}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

        endpoint_id = f"e-{generate_ulid()}"
        now = get_current_time_millis()

        # Auto-create experiment if usage_tracking is enabled and no experiment_id provided
        if usage_tracking and experiment_id is None:
            exp_name = f"gateway/{name}"
            existing_exp = self.get_experiment_by_name(exp_name)
            experiment_id = (
                existing_exp.experiment_id if existing_exp else self.create_experiment(exp_name)
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_TYPE, "GATEWAY")
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_ID, endpoint_id)
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_IS_GATEWAY, "true")
            )

        # Build META item
        meta_item: dict[str, Any] = {
            "PK": f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            "SK": SK_GW_META,
            "name": name,
            "created_at": now,
            "last_updated_at": now,
            "workspace": self._workspace,
            "usage_tracking": usage_tracking,
            "tags": {},
            # GSI projections
            "gsi2pk": f"{GSI2_GW_ENDPOINTS_PREFIX}{self._workspace}",
            "gsi2sk": endpoint_id,
            "gsi3pk": f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}",
            "gsi3sk": endpoint_id,
        }
        if routing_strategy is not None:
            meta_item["routing_strategy"] = routing_strategy.value
        if fallback_config is not None:
            meta_item["fallback_config"] = {
                "strategy": fallback_config.strategy.value if fallback_config.strategy else None,
                "max_attempts": fallback_config.max_attempts,
            }
        if experiment_id is not None:
            meta_item["experiment_id"] = experiment_id
        if created_by is not None:
            meta_item["created_by"] = created_by
            meta_item["last_updated_by"] = created_by

        # Build mapping items
        mapping_items = []
        for config in model_configs:
            mapping_id = f"m-{generate_ulid()}"
            mapping_items.append(
                self._build_mapping_item(endpoint_id, config, mapping_id, now, created_by)
            )

        # Write META
        self._table.put_item(meta_item)

        # Batch write mapping items
        if mapping_items:
            self._table.batch_write(mapping_items)

        # Return the full entity
        all_items = [meta_item] + mapping_items
        return self._endpoint_items_to_entity(endpoint_id, all_items)

    def get_gateway_endpoint(
        self,
        endpoint_id: str | None = None,
        name: str | None = None,
    ) -> GatewayEndpoint:
        if (endpoint_id is None) == (name is None):
            raise MlflowException(
                "Exactly one of `endpoint_id` or `name` must be specified",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if name:
            # Resolve name to endpoint_id via GSI3
            results = self._table.query(
                pk=f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}",
                index_name="gsi3",
                limit=1,
            )
            if not results:
                raise MlflowException(
                    f"GatewayEndpoint not found (name='{name}')",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            endpoint_id = results[0]["gsi3sk"]

        assert (
            endpoint_id is not None
        )  # guaranteed by the (endpoint_id is None) == (name is None) check
        items = self._get_endpoint_items(endpoint_id)
        if not items:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return self._endpoint_items_to_entity(endpoint_id, items)

    def list_gateway_endpoints(
        self,
        provider: str | None = None,
        secret_id: str | None = None,
    ) -> list[GatewayEndpoint]:
        if secret_id:
            # Targeted path: find model_defs by secret -> find endpoints by model_def -> deduplicate
            md_items = self._table.query(
                pk=f"{GSI4_GW_MODELDEF_SECRET_PREFIX}{secret_id}",
                index_name="gsi4",
            )
            endpoint_ids: set[str] = set()
            for md_item in md_items:
                md_id = md_item["PK"].removeprefix(PK_GW_MODELDEF_PREFIX)
                ep_items = self._table.query(
                    pk=f"{GSI5_GW_ENDPOINT_MODELDEF_PREFIX}{md_id}",
                    index_name="gsi5",
                )
                for ep_item in ep_items:
                    endpoint_ids.add(ep_item["gsi5sk"])

            endpoints = []
            for ep_id in endpoint_ids:
                items = self._get_endpoint_items(ep_id)
                if items:
                    endpoints.append(self._endpoint_items_to_entity(ep_id, items))
            return endpoints

        # Default path: list all endpoints via GSI2
        gsi2_items = self._table.query(
            pk=f"{GSI2_GW_ENDPOINTS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )

        endpoints = []
        for gsi2_item in gsi2_items:
            ep_id = gsi2_item["gsi2sk"]
            items = self._get_endpoint_items(ep_id)
            if items:
                endpoints.append(self._endpoint_items_to_entity(ep_id, items))

        if provider is not None:
            endpoints = [
                ep
                for ep in endpoints
                if any(
                    m.model_definition and m.model_definition.provider == provider
                    for m in ep.model_mappings
                )
            ]

        return endpoints

    def update_gateway_endpoint(
        self,
        endpoint_id: str,
        name: str | None = None,
        updated_by: str | None = None,
        routing_strategy: RoutingStrategy | None = None,
        fallback_config: FallbackConfig | None = None,
        model_configs: list[GatewayEndpointModelConfig] | None = None,
        experiment_id: str | None = None,
        usage_tracking: bool | None = None,
    ) -> GatewayEndpoint:
        # Get existing endpoint items
        items = self._get_endpoint_items(endpoint_id)
        meta = next((i for i in items if i["SK"] == SK_GW_META), None)
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()
        updates: dict[str, Any] = {"last_updated_at": now}
        removes: list[str] = []

        if name is not None:
            # Check new name uniqueness
            existing = self._table.query(
                pk=f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}",
                index_name="gsi3",
                limit=1,
            )
            if existing and existing[0]["gsi3sk"] != endpoint_id:
                raise MlflowException(
                    f"Endpoint with name '{name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                )
            updates["name"] = name
            updates["gsi3pk"] = f"{GSI3_GW_ENDPOINT_NAME_PREFIX}{self._workspace}#{name}"

        # Handle usage_tracking update
        if usage_tracking is not None:
            updates["usage_tracking"] = usage_tracking

        # Auto-create experiment if usage_tracking is enabled and no experiment_id provided
        if usage_tracking and experiment_id is None and meta.get("experiment_id") is None:
            endpoint_name = name if name is not None else meta.get("name")
            exp_name = f"gateway/{endpoint_name}"
            existing_exp = self.get_experiment_by_name(exp_name)
            experiment_id = (
                existing_exp.experiment_id if existing_exp else self.create_experiment(exp_name)
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_TYPE, "GATEWAY")
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_SOURCE_ID, endpoint_id)
            )
            self.set_experiment_tag(
                experiment_id, ExperimentTag(MLFLOW_EXPERIMENT_IS_GATEWAY, "true")
            )

        if experiment_id is not None:
            updates["experiment_id"] = experiment_id

        if routing_strategy is not None:
            updates["routing_strategy"] = routing_strategy.value

        if updated_by is not None:
            updates["last_updated_by"] = updated_by

        # Replace model configs if provided (full replacement)
        if model_configs is not None:
            # Validate all model definitions exist
            for config in model_configs:
                md_item = self._get_model_def_item(config.model_definition_id)
                if md_item is None:
                    raise MlflowException(
                        f"Model definition '{config.model_definition_id}' not found",
                        error_code=RESOURCE_DOES_NOT_EXIST,
                    )

            # Delete existing mapping items
            existing_mappings = [i for i in items if i["SK"].startswith(SK_GW_MAP_PREFIX)]
            if existing_mappings:
                self._table.batch_delete(
                    [{"PK": m["PK"], "SK": m["SK"]} for m in existing_mappings]
                )

            # Write new mapping items
            new_mapping_items = []
            for config in model_configs:
                mapping_id = f"m-{generate_ulid()}"
                new_mapping_items.append(
                    self._build_mapping_item(endpoint_id, config, mapping_id, now, updated_by)
                )
            if new_mapping_items:
                self._table.batch_write(new_mapping_items)

            # Update fallback_config with new model config info
            fallback_model_def_ids = [
                config.model_definition_id
                for config in model_configs
                if config.linkage_type == GatewayModelLinkageType.FALLBACK
            ]
            if fallback_config or fallback_model_def_ids:
                updates["fallback_config"] = {
                    "strategy": fallback_config.strategy.value
                    if fallback_config and fallback_config.strategy
                    else None,
                    "max_attempts": fallback_config.max_attempts if fallback_config else None,
                }
            else:
                # No fallback models and no explicit fallback_config, clear it
                updates["fallback_config"] = {
                    "strategy": None,
                    "max_attempts": None,
                }

        elif fallback_config is not None:
            # Update fallback_config without replacing model configs
            updates["fallback_config"] = {
                "strategy": fallback_config.strategy.value if fallback_config.strategy else None,
                "max_attempts": fallback_config.max_attempts,
            }

        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates=updates,
            removes=removes if removes else None,
        )

        self._invalidate_secret_cache()

        # Re-fetch to return the full updated entity
        updated_items = self._get_endpoint_items(endpoint_id)
        return self._endpoint_items_to_entity(endpoint_id, updated_items)

    def delete_gateway_endpoint(self, endpoint_id: str) -> None:
        # Query all items in partition
        items = self._get_endpoint_items(endpoint_id)
        if not items:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Batch delete all items (META, mappings, bindings, tags)
        self._table.batch_delete([{"PK": item["PK"], "SK": item["SK"]} for item in items])
        self._invalidate_secret_cache()

    def attach_model_to_endpoint(
        self,
        endpoint_id: str,
        model_config: GatewayEndpointModelConfig,
        created_by: str | None = None,
    ) -> GatewayEndpointModelMapping:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        # Verify model definition exists
        md_item = self._get_model_def_item(model_config.model_definition_id)
        if md_item is None:
            raise MlflowException(
                f"Model definition '{model_config.model_definition_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        mapping_id = f"m-{generate_ulid()}"
        now = get_current_time_millis()

        mapping_item = self._build_mapping_item(
            endpoint_id, model_config, mapping_id, now, created_by
        )

        # Conditional put for atomic uniqueness
        from botocore.exceptions import ClientError

        try:
            self._table.put_item(mapping_item, condition="attribute_not_exists(SK)")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise MlflowException(
                    f"Model definition '{model_config.model_definition_id}' is already attached to "
                    f"endpoint '{endpoint_id}'",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e
            raise

        # Update endpoint META last_updated_at
        update_fields: dict[str, Any] = {"last_updated_at": now}
        if created_by:
            update_fields["last_updated_by"] = created_by
        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates=update_fields,
        )

        self._invalidate_secret_cache()

        # Return the mapping entity
        model_def = self._model_def_item_to_entity(md_item)
        return GatewayEndpointModelMapping(
            mapping_id=mapping_id,
            endpoint_id=endpoint_id,
            model_definition_id=model_config.model_definition_id,
            model_definition=model_def,
            weight=float(model_config.weight),
            linkage_type=model_config.linkage_type,
            fallback_order=model_config.fallback_order,
            created_at=now,
            created_by=created_by,
        )

    def detach_model_from_endpoint(
        self,
        endpoint_id: str,
        model_definition_id: str,
        linkage_type: str | None = None,
    ) -> None:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        if linkage_type:
            # Deterministic SK — direct delete
            sk = f"{SK_GW_MAP_PREFIX}{model_definition_id}#{linkage_type}"
            existing = self._table.get_item(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
                sk=sk,
            )
            if existing is None:
                raise MlflowException(
                    f"Model definition '{model_definition_id}' is not attached to "
                    f"endpoint '{endpoint_id}' with linkage type '{linkage_type}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            self._table.delete_item(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
                sk=sk,
            )
        else:
            # Query all linkage variants for this model_def
            mapping_items = self._table.query(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
                sk_prefix=f"{SK_GW_MAP_PREFIX}{model_definition_id}#",
            )
            if not mapping_items:
                raise MlflowException(
                    f"Model definition '{model_definition_id}' is not attached to "
                    f"endpoint '{endpoint_id}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            self._table.batch_delete([{"PK": m["PK"], "SK": m["SK"]} for m in mapping_items])

        # Update endpoint META last_updated_at
        now = get_current_time_millis()
        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates={"last_updated_at": now},
        )
        self._invalidate_secret_cache()

    def create_endpoint_binding(
        self,
        endpoint_id: str,
        resource_type: str,
        resource_id: str,
        created_by: str | None = None,
    ) -> GatewayEndpointBinding:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()

        item: dict[str, Any] = {
            "PK": f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            "SK": f"{SK_GW_BIND_PREFIX}{resource_type}#{resource_id}",
            "endpoint_id": endpoint_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "created_at": now,
            "last_updated_at": now,
            # GSI2 projection for reverse lookup (AP25)
            "gsi2pk": f"{GSI2_GW_BIND_PREFIX}{resource_type}#{resource_id}",
            "gsi2sk": endpoint_id,
        }
        if created_by is not None:
            item["created_by"] = created_by
            item["last_updated_by"] = created_by

        self._table.put_item(item)
        self._invalidate_secret_cache()

        return GatewayEndpointBinding(
            endpoint_id=endpoint_id,
            resource_type=GatewayResourceType(resource_type),
            resource_id=resource_id,
            created_at=now,
            last_updated_at=now,
            created_by=created_by,
            last_updated_by=created_by,
        )

    def delete_endpoint_binding(
        self,
        endpoint_id: str,
        resource_type: str,
        resource_id: str,
    ) -> None:
        # Verify binding exists
        sk = f"{SK_GW_BIND_PREFIX}{resource_type}#{resource_id}"
        existing = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=sk,
        )
        if existing is None:
            raise MlflowException(
                f"GatewayEndpointBinding not found (endpoint_id='{endpoint_id}', "
                f"resource_type='{resource_type}', resource_id='{resource_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        self._table.delete_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=sk,
        )
        self._invalidate_secret_cache()

    def _binding_item_to_entity(self, item: dict[str, Any]) -> GatewayEndpointBinding:
        """Convert a raw DynamoDB binding item to a GatewayEndpointBinding entity."""
        return GatewayEndpointBinding(
            endpoint_id=item["endpoint_id"],
            resource_type=GatewayResourceType(item["resource_type"]),
            resource_id=item["resource_id"],
            created_at=int(item["created_at"]),
            last_updated_at=int(item["last_updated_at"]),
            created_by=item.get("created_by"),
            last_updated_by=item.get("last_updated_by"),
            display_name=item.get("display_name"),
        )

    def list_endpoint_bindings(
        self,
        endpoint_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> list[GatewayEndpointBinding]:
        if endpoint_id is not None:
            # Direct query within endpoint partition
            sk_prefix = SK_GW_BIND_PREFIX
            if resource_type is not None:
                sk_prefix = f"{SK_GW_BIND_PREFIX}{resource_type}#"
                if resource_id is not None:
                    sk_prefix = f"{SK_GW_BIND_PREFIX}{resource_type}#{resource_id}"

            items = self._table.query(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
                sk_prefix=sk_prefix,
            )
            return [self._binding_item_to_entity(item) for item in items]

        if resource_type is not None and resource_id is not None:
            # Reverse lookup via GSI2
            gsi2_items = self._table.query(
                pk=f"{GSI2_GW_BIND_PREFIX}{resource_type}#{resource_id}",
                index_name="gsi2",
            )
            bindings = []
            for gsi2_item in gsi2_items:
                ep_id = gsi2_item["gsi2sk"]
                sk = f"{SK_GW_BIND_PREFIX}{resource_type}#{resource_id}"
                bind_item = self._table.get_item(
                    pk=f"{PK_GW_ENDPOINT_PREFIX}{ep_id}",
                    sk=sk,
                )
                if bind_item:
                    bindings.append(self._binding_item_to_entity(bind_item))
            return bindings

        # No endpoint_id and no resource filter: list all endpoints, query bindings per endpoint
        gsi2_items = self._table.query(
            pk=f"{GSI2_GW_ENDPOINTS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )
        bindings = []
        for gsi2_item in gsi2_items:
            ep_id = gsi2_item["gsi2sk"]
            bind_items = self._table.query(
                pk=f"{PK_GW_ENDPOINT_PREFIX}{ep_id}",
                sk_prefix=SK_GW_BIND_PREFIX,
            )
            bindings.extend(self._binding_item_to_entity(item) for item in bind_items)
        return bindings

    def set_gateway_endpoint_tag(
        self,
        endpoint_id: str,
        tag: GatewayEndpointTag,
    ) -> None:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()

        # Write the tag item (upsert)
        tag_item: dict[str, Any] = {
            "PK": f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            "SK": f"{SK_GW_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(tag_item)

        # Update denormalized tags on META
        tags_dict = dict(meta.get("tags", {}))
        tags_dict[tag.key] = tag.value
        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates={"tags": tags_dict, "last_updated_at": now},
        )

    def delete_gateway_endpoint_tag(
        self,
        endpoint_id: str,
        key: str,
    ) -> None:
        # Verify endpoint exists
        meta = self._table.get_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
        )
        if meta is None:
            raise MlflowException(
                f"GatewayEndpoint not found (endpoint_id='{endpoint_id}')",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        now = get_current_time_millis()

        # Delete the tag item (no-op if it doesn't exist)
        self._table.delete_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=f"{SK_GW_TAG_PREFIX}{key}",
        )

        # Update denormalized tags on META
        tags_dict = dict(meta.get("tags", {}))
        tags_dict.pop(key, None)
        self._table.update_item(
            pk=f"{PK_GW_ENDPOINT_PREFIX}{endpoint_id}",
            sk=SK_GW_META,
            updates={"tags": tags_dict, "last_updated_at": now},
        )
