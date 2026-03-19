"""Field-level comparison policies for store entity types.

Each entity type maps field names to a comparison category:
- MUST_MATCH: values must be exactly equal
- TYPE_MUST_MATCH: Python types must match, values may differ
- IGNORE: skip comparison entirely

Unknown fields default to MUST_MATCH (fail-safe).
"""

from enum import Enum


class FieldPolicy(Enum):
    MUST_MATCH = "must_match"
    TYPE_MUST_MATCH = "type_must_match"
    IGNORE = "ignore"


MUST_MATCH = FieldPolicy.MUST_MATCH
TYPE_MUST_MATCH = FieldPolicy.TYPE_MUST_MATCH
IGNORE = FieldPolicy.IGNORE

DEFAULT_POLICY = MUST_MATCH


EXPERIMENT = {
    "experiment_id": IGNORE,
    "name": MUST_MATCH,
    "artifact_location": IGNORE,
    "lifecycle_stage": MUST_MATCH,
    "tags": MUST_MATCH,
    "creation_time": TYPE_MUST_MATCH,
    "last_update_time": TYPE_MUST_MATCH,
}

RUN_INFO = {
    "run_id": IGNORE,
    "run_uuid": IGNORE,
    "experiment_id": IGNORE,
    "user_id": MUST_MATCH,
    "status": MUST_MATCH,
    "start_time": TYPE_MUST_MATCH,
    "end_time": TYPE_MUST_MATCH,
    "artifact_uri": IGNORE,
    "lifecycle_stage": MUST_MATCH,
    "run_name": MUST_MATCH,
}

RUN_DATA = {
    "metrics": MUST_MATCH,
    "params": MUST_MATCH,
    "tags": MUST_MATCH,
}

METRIC = {
    "key": MUST_MATCH,
    "value": MUST_MATCH,
    "timestamp": TYPE_MUST_MATCH,
    "step": MUST_MATCH,
}

PARAM = {
    "key": MUST_MATCH,
    "value": MUST_MATCH,
}

TAG = {
    "key": MUST_MATCH,
    "value": MUST_MATCH,
}

REGISTERED_MODEL = {
    "name": MUST_MATCH,
    "creation_timestamp": TYPE_MUST_MATCH,
    "last_updated_timestamp": TYPE_MUST_MATCH,
    "description": MUST_MATCH,
    "latest_versions": MUST_MATCH,
    "tags": MUST_MATCH,
    "aliases": MUST_MATCH,
}

MODEL_VERSION = {
    "name": MUST_MATCH,
    "version": MUST_MATCH,
    "creation_timestamp": TYPE_MUST_MATCH,
    "last_updated_timestamp": TYPE_MUST_MATCH,
    "description": MUST_MATCH,
    "user_id": MUST_MATCH,
    "current_stage": MUST_MATCH,
    "source": MUST_MATCH,
    "run_id": IGNORE,
    "status": MUST_MATCH,
    "status_message": MUST_MATCH,
    "tags": MUST_MATCH,
    "run_link": MUST_MATCH,
    "aliases": MUST_MATCH,
}

WORKSPACE = {
    "name": MUST_MATCH,
    "description": MUST_MATCH,
    "creation_time": TYPE_MUST_MATCH,
    "last_update_time": TYPE_MUST_MATCH,
}

TRACE_INFO = {
    "request_id": IGNORE,
    "experiment_id": IGNORE,
    "timestamp_ms": TYPE_MUST_MATCH,
    "execution_time_ms": TYPE_MUST_MATCH,
    "status": MUST_MATCH,
    "request_metadata": MUST_MATCH,
    "tags": MUST_MATCH,
}
