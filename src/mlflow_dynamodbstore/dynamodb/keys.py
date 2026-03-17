"""DynamoDB key builder functions for all entity types.

All functions return DynamoDB-formatted attribute dicts using the
``{"S": "value"}`` / ``{"N": "123"}`` wire format.
"""

from __future__ import annotations

from typing import Any

from mlflow_dynamodbstore.dynamodb.schema import (
    GSI1_PK,
    GSI1_RUN_PREFIX,
    GSI1_SK,
    GSI2_EXPERIMENTS_PREFIX,
    GSI2_MODELS_PREFIX,
    GSI2_PK,
    GSI2_SK,
    GSI3_EXP_NAME_PREFIX,
    GSI3_MODEL_NAME_PREFIX,
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
    PK_MODEL_PREFIX,
    PK_USER_PREFIX,
    PK_WORKSPACE_PREFIX,
    SK_EXPERIMENT_META,
    SK_EXPERIMENT_TAG_PREFIX,
    SK_METRIC_HISTORY_PREFIX,
    SK_METRIC_PREFIX,
    SK_MODEL_ALIAS_PREFIX,
    SK_MODEL_META,
    SK_MODEL_TAG_PREFIX,
    SK_PARAM_PREFIX,
    SK_RUN_PREFIX,
    SK_TAG_PREFIX,
    SK_VERSION_PREFIX,
    SK_WORKSPACE_META,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP_PAD_WIDTH = 20
_STEP_NEG_COMPLEMENT = 9_999_999_999_999_999_999


def rev(s: str) -> str:
    """Return *s* reversed."""
    return s[::-1]


def pad_step(step: int) -> str:
    """Return a sortable zero-padded step string with sign prefix.

    Positive / zero steps:  ``P#<20-digit zero-padded step>``
    Negative steps:         ``N#<20-digit complement>``  so that ascending
                            sort produces correct order for negative steps.
    """
    if step >= 0:
        return f"P#{step:0{_STEP_PAD_WIDTH}d}"
    complement = _STEP_NEG_COMPLEMENT + step
    return f"N#{complement:0{_STEP_PAD_WIDTH}d}"


# ---------------------------------------------------------------------------
# Experiment key builders
# ---------------------------------------------------------------------------


def experiment_pk(exp_id: str) -> dict[str, Any]:
    return {"PK": {"S": f"{PK_EXPERIMENT_PREFIX}{exp_id}"}}


def experiment_meta_sk() -> dict[str, Any]:
    return {"SK": {"S": SK_EXPERIMENT_META}}


def experiment_meta_lsi(
    *,
    lifecycle: str,
    ulid: str,
    last_update_time: int,
    name: str,
) -> dict[str, Any]:
    return {
        LSI1_SK: {"S": f"{lifecycle}#{ulid}"},
        LSI2_SK: {"N": str(last_update_time)},
        LSI3_SK: {"S": name},
        LSI4_SK: {"S": rev(name)},
    }


def experiment_gsi2(*, workspace: str, lifecycle: str, ulid: str) -> dict[str, Any]:
    return {
        GSI2_PK: {"S": f"{GSI2_EXPERIMENTS_PREFIX}{workspace}#{lifecycle}"},
        GSI2_SK: {"S": ulid},
    }


def experiment_gsi3(*, workspace: str, name: str, exp_id: str) -> dict[str, Any]:
    return {
        GSI3_PK: {"S": f"{GSI3_EXP_NAME_PREFIX}{workspace}#{name}"},
        GSI3_SK: {"S": exp_id},
    }


def experiment_gsi5(*, workspace: str, name: str, exp_id: str) -> dict[str, Any]:
    return {
        GSI5_PK: {"S": f"{GSI5_EXP_NAMES_PREFIX}{workspace}"},
        GSI5_SK: {"S": f"{name}#{exp_id}"},
    }


def experiment_tag_sk(key: str) -> dict[str, Any]:
    return {"SK": {"S": f"{SK_EXPERIMENT_TAG_PREFIX}{key}"}}


# ---------------------------------------------------------------------------
# Run key builders
# ---------------------------------------------------------------------------


def run_pk(exp_id: str) -> dict[str, Any]:
    """Runs live in the same partition as their experiment."""
    return experiment_pk(exp_id)


def run_meta_sk(run_ulid: str) -> dict[str, Any]:
    return {"SK": {"S": f"{SK_RUN_PREFIX}{run_ulid}"}}


def run_meta_lsi(
    *,
    lifecycle: str,
    ulid: str,
    status: str,
    run_name: str,
) -> dict[str, Any]:
    return {
        LSI1_SK: {"S": f"{lifecycle}#{ulid}"},
        LSI3_SK: {"S": f"{status}#{ulid}"},
        LSI4_SK: {"S": run_name.lower()},
    }


def run_gsi1(*, run_id: str, exp_id: str) -> dict[str, Any]:
    return {
        GSI1_PK: {"S": f"{GSI1_RUN_PREFIX}{run_id}"},
        GSI1_SK: {"S": exp_id},
    }


def run_tag_sk(run_ulid: str, key: str) -> dict[str, Any]:
    return {"SK": {"S": f"{SK_RUN_PREFIX}{run_ulid}{SK_TAG_PREFIX}{key}"}}


def run_param_sk(run_ulid: str, key: str) -> dict[str, Any]:
    return {"SK": {"S": f"{SK_RUN_PREFIX}{run_ulid}{SK_PARAM_PREFIX}{key}"}}


# ---------------------------------------------------------------------------
# Metric key builders
# ---------------------------------------------------------------------------


def metric_latest_sk(run_ulid: str, key: str) -> dict[str, Any]:
    return {"SK": {"S": f"{SK_RUN_PREFIX}{run_ulid}{SK_METRIC_PREFIX}{key}"}}


def metric_history_sk(
    *,
    run_ulid: str,
    key: str,
    step: int,
    timestamp: int,
) -> dict[str, Any]:
    padded = pad_step(step)
    return {
        "SK": {
            "S": (f"{SK_RUN_PREFIX}{run_ulid}{SK_METRIC_HISTORY_PREFIX}{key}#{padded}#{timestamp}")
        }
    }


# ---------------------------------------------------------------------------
# Model (Registered Model) key builders
# ---------------------------------------------------------------------------


def model_pk(model_ulid: str) -> dict[str, Any]:
    return {"PK": {"S": f"{PK_MODEL_PREFIX}{model_ulid}"}}


def model_meta_sk() -> dict[str, Any]:
    return {"SK": {"S": SK_MODEL_META}}


def model_meta_lsi(*, last_update_time: int, name: str) -> dict[str, Any]:
    return {
        LSI2_SK: {"N": str(last_update_time)},
        LSI3_SK: {"S": name},
        LSI4_SK: {"S": rev(name)},
    }


def model_gsi2(*, workspace: str, last_update_time: int, name: str) -> dict[str, Any]:
    return {
        GSI2_PK: {"S": f"{GSI2_MODELS_PREFIX}{workspace}"},
        GSI2_SK: {"S": f"{last_update_time}#{name}"},
    }


def model_gsi3(*, workspace: str, name: str, model_id: str) -> dict[str, Any]:
    return {
        GSI3_PK: {"S": f"{GSI3_MODEL_NAME_PREFIX}{workspace}#{name}"},
        GSI3_SK: {"S": model_id},
    }


def model_tag_sk(key: str) -> dict[str, Any]:
    return {"SK": {"S": f"{SK_MODEL_TAG_PREFIX}{key}"}}


def model_alias_sk(alias: str) -> dict[str, Any]:
    return {"SK": {"S": f"{SK_MODEL_ALIAS_PREFIX}{alias}"}}


# ---------------------------------------------------------------------------
# Model Version key builders
# ---------------------------------------------------------------------------


def version_meta_sk(padded_ver: str) -> dict[str, Any]:
    return {"SK": {"S": f"{SK_VERSION_PREFIX}{padded_ver}"}}


def version_meta_lsi(
    *,
    creation_time: int,
    last_update_time: int,
    stage: str,
    padded_ver: str,
    source_path: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        LSI1_SK: {"S": f"{creation_time:020d}"},
        LSI2_SK: {"N": str(last_update_time)},  # lsi2sk is Number in schema
        LSI3_SK: {"S": f"{stage}#{padded_ver}"},
        LSI4_SK: {"S": source_path},
        LSI5_SK: {"S": f"{run_id}#{padded_ver}"},
    }


# ---------------------------------------------------------------------------
# Workspace / User key builders
# ---------------------------------------------------------------------------


def workspace_pk(workspace_name: str) -> dict[str, Any]:
    return {"PK": {"S": f"{PK_WORKSPACE_PREFIX}{workspace_name}"}}


def workspace_meta_sk() -> dict[str, Any]:
    return {"SK": {"S": SK_WORKSPACE_META}}


def user_pk(username: str) -> dict[str, Any]:
    return {"PK": {"S": f"{PK_USER_PREFIX}{username}"}}
