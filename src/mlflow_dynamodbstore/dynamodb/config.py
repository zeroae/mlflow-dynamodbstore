"""ConfigReader: reads and caches denormalization and FTS trigram configuration from DynamoDB."""

from __future__ import annotations

import fnmatch
import os

from mlflow_dynamodbstore.dynamodb.schema import (
    CONFIG_DENORMALIZE_TAGS,
    CONFIG_FTS_TRIGRAM_FIELDS,
    CONFIG_TTL_POLICY,
    PK_CONFIG,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable

# Fields that always get trigrams regardless of config
_ALWAYS_TRIGRAM_FIELDS: frozenset[str] = frozenset({"experiment_name", "run_name", "model_name"})

# Default denormalize patterns (always included)
_DEFAULT_DENORMALIZE_PATTERNS: list[str] = ["mlflow.*"]

# ENV var name for overriding denormalize patterns on startup
_ENV_DENORMALIZE_TAGS = "MLFLOW_DYNAMODB_DENORMALIZE_TAGS"

# SK suffix for per-experiment denormalize config
_SK_EXP_DENORMALIZE_PREFIX = "DENORMALIZE_TAGS#EXP#"


class ConfigReader:
    """Reads CONFIG items from DynamoDB, caches in memory.

    Provides:
    - should_denormalize(experiment_id, tag_key): matches tag_key against patterns
    - should_trigram(field_type): entity name fields always return True; others are configurable
    - reconcile(): reads env vars and merges with defaults, persisting to DynamoDB
    """

    def __init__(self, table: DynamoDBTable) -> None:
        self._table = table
        # In-memory caches
        self._denormalize_patterns: list[str] | None = None
        self._fts_trigram_fields: list[str] | None = None
        self._ttl_policy: dict[str, int] | None = None
        # Per-experiment denormalize pattern cache: experiment_id -> list[str]
        self._exp_denormalize_patterns: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Denormalize patterns
    # ------------------------------------------------------------------

    def get_denormalize_patterns(self) -> list[str]:
        """Return the global denormalize tag patterns, loading from DynamoDB if needed."""
        if self._denormalize_patterns is None:
            self._denormalize_patterns = self._load_denormalize_patterns()
        return self._denormalize_patterns

    def _load_denormalize_patterns(self) -> list[str]:
        """Load global denormalize patterns from DynamoDB, falling back to defaults."""
        item = self._table.get_item(pk=PK_CONFIG, sk=CONFIG_DENORMALIZE_TAGS)
        if item and "patterns" in item:
            patterns: list[str] = list(item["patterns"])
        else:
            patterns = list(_DEFAULT_DENORMALIZE_PATTERNS)
        # Always ensure mlflow.* is present
        if "mlflow.*" not in patterns:
            patterns.insert(0, "mlflow.*")
        return patterns

    def set_denormalize_patterns(self, patterns: list[str]) -> None:
        """Persist global denormalize patterns to DynamoDB and update cache."""
        # Always ensure mlflow.* is present
        merged = list(patterns)
        if "mlflow.*" not in merged:
            merged.insert(0, "mlflow.*")
        self._table.put_item(
            {
                "PK": PK_CONFIG,
                "SK": CONFIG_DENORMALIZE_TAGS,
                "patterns": merged,
            }
        )
        self._denormalize_patterns = merged

    def set_experiment_denormalize_patterns(self, experiment_id: str, patterns: list[str]) -> None:
        """Persist per-experiment denormalize patterns to DynamoDB and update cache."""
        sk = f"{_SK_EXP_DENORMALIZE_PREFIX}{experiment_id}"
        self._table.put_item(
            {
                "PK": PK_CONFIG,
                "SK": sk,
                "patterns": list(patterns),
            }
        )
        self._exp_denormalize_patterns[experiment_id] = list(patterns)

    def get_experiment_denormalize_patterns(self, experiment_id: str) -> list[str]:
        """Return per-experiment denormalize patterns (not merged with global)."""
        if experiment_id not in self._exp_denormalize_patterns:
            sk = f"{_SK_EXP_DENORMALIZE_PREFIX}{experiment_id}"
            item = self._table.get_item(pk=PK_CONFIG, sk=sk)
            if item and "patterns" in item:
                self._exp_denormalize_patterns[experiment_id] = list(item["patterns"])
            else:
                self._exp_denormalize_patterns[experiment_id] = []
        return self._exp_denormalize_patterns[experiment_id]

    def get_effective_denormalize_patterns(self, experiment_id: str) -> list[str]:
        """Return merged global + per-experiment denormalize patterns."""
        global_patterns = self.get_denormalize_patterns()
        exp_patterns = self.get_experiment_denormalize_patterns(experiment_id)
        # Merge, preserving order, deduplicating
        merged: list[str] = list(global_patterns)
        for p in exp_patterns:
            if p not in merged:
                merged.append(p)
        return merged

    def should_denormalize(self, experiment_id: str | None, tag_key: str) -> bool:
        """Return True if tag_key matches any effective denormalize pattern."""
        if experiment_id is not None:
            patterns = self.get_effective_denormalize_patterns(experiment_id)
        else:
            patterns = self.get_denormalize_patterns()
        return any(fnmatch.fnmatch(tag_key, pattern) for pattern in patterns)

    # ------------------------------------------------------------------
    # FTS trigram fields
    # ------------------------------------------------------------------

    def get_fts_trigram_fields(self) -> list[str]:
        """Return configurable FTS trigram fields (not including always-trigram fields)."""
        if self._fts_trigram_fields is None:
            self._fts_trigram_fields = self._load_fts_trigram_fields()
        return self._fts_trigram_fields

    def _load_fts_trigram_fields(self) -> list[str]:
        """Load FTS trigram fields from DynamoDB, falling back to empty list."""
        item = self._table.get_item(pk=PK_CONFIG, sk=CONFIG_FTS_TRIGRAM_FIELDS)
        if item and "fields" in item:
            return list(item["fields"])
        return []

    def set_fts_trigram_fields(self, fields: list[str]) -> None:
        """Persist FTS trigram fields to DynamoDB and update cache."""
        self._table.put_item(
            {
                "PK": PK_CONFIG,
                "SK": CONFIG_FTS_TRIGRAM_FIELDS,
                "fields": list(fields),
            }
        )
        self._fts_trigram_fields = list(fields)

    def should_trigram(self, field_type: str) -> bool:
        """Return True if field_type should have trigram indexing.

        Entity name fields (experiment_name, run_name, model_name) always return True.
        Other fields are checked against the configured FTS trigram fields list.
        """
        if field_type in _ALWAYS_TRIGRAM_FIELDS:
            return True
        return field_type in self.get_fts_trigram_fields()

    # ------------------------------------------------------------------
    # TTL policy
    # ------------------------------------------------------------------

    _DEFAULT_TTL_POLICY: dict[str, int] = {
        "trace_retention_days": 30,
        "soft_deleted_retention_days": 90,
        "metric_history_retention_days": 365,
    }

    def get_ttl_policy(self) -> dict[str, int]:
        """Return the TTL policy, loading from DynamoDB if needed."""
        if self._ttl_policy is None:
            self._ttl_policy = self._load_ttl_policy()
        return self._ttl_policy

    def _load_ttl_policy(self) -> dict[str, int]:
        """Load TTL policy from DynamoDB, falling back to defaults."""
        item = self._table.get_item(pk=PK_CONFIG, sk=CONFIG_TTL_POLICY)
        policy = dict(self._DEFAULT_TTL_POLICY)
        if item:
            for key in policy:
                if key in item:
                    policy[key] = int(item[key])
        return policy

    # ------------------------------------------------------------------
    # Reconcile from environment
    # ------------------------------------------------------------------

    def reconcile(self) -> None:
        """Read env vars and merge with defaults, persisting updated config to DynamoDB."""
        env_value = os.environ.get(_ENV_DENORMALIZE_TAGS)
        if env_value:
            # Parse comma-separated patterns from env var
            env_patterns = [p.strip() for p in env_value.split(",") if p.strip()]
            # Merge with current patterns, ensuring mlflow.* is always present
            current = self.get_denormalize_patterns()
            merged: list[str] = list(current)
            for p in env_patterns:
                if p not in merged:
                    merged.append(p)
            if "mlflow.*" not in merged:
                merged.insert(0, "mlflow.*")
            self.set_denormalize_patterns(merged)
        else:
            # Ensure defaults are persisted (idempotent)
            patterns = self.get_denormalize_patterns()
            self.set_denormalize_patterns(patterns)
