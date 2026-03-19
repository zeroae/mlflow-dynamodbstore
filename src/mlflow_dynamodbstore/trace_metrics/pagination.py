"""DynamoDB cache and page token management for query_trace_metrics."""

from __future__ import annotations

import hashlib
import json
import time
from base64 import b64decode, b64encode
from typing import Any

from mlflow.entities.trace_metrics import MetricAggregation, MetricDataPoint, MetricViewType

CACHE_TTL_SECONDS = 900  # 15 minutes


def compute_query_hash(
    experiment_ids: list[str],
    view_type: MetricViewType,
    metric_name: str,
    aggregations: list[MetricAggregation],
    dimensions: list[str] | None,
    filters: list[str] | None,
    time_interval_seconds: int | None,
    start_time_ms: int | None,
    end_time_ms: int | None,
) -> str:
    key_parts = {
        "experiment_ids": sorted(experiment_ids),
        "view_type": str(view_type),
        "metric_name": metric_name,
        "aggregations": sorted(str(a) for a in aggregations),
        "dimensions": sorted(dimensions) if dimensions else [],
        "filters": sorted(filters) if filters else [],
        "time_interval_seconds": time_interval_seconds,
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
    }
    canonical = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def encode_page_token(query_hash: str, offset: int) -> str:
    return b64encode(json.dumps({"query_hash": query_hash, "offset": offset}).encode()).decode()


def decode_page_token(token: str) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(b64decode(token).decode())
    return result


def cache_put(table: Any, query_hash: str, data_points: list[MetricDataPoint]) -> None:
    from mlflow_dynamodbstore.dynamodb.schema import PK_TMCACHE_PREFIX, SK_TMCACHE_RESULT

    serialized = [
        {
            "metric_name": dp.metric_name,
            "dimensions": dp.dimensions,
            "values": dp.values,
        }
        for dp in data_points
    ]
    table.put_item(
        {
            "PK": f"{PK_TMCACHE_PREFIX}{query_hash}",
            "SK": SK_TMCACHE_RESULT,
            "data": json.dumps(serialized),
            "ttl": int(time.time()) + CACHE_TTL_SECONDS,
        }
    )


def cache_get(table: Any, query_hash: str) -> list[MetricDataPoint] | None:
    from mlflow_dynamodbstore.dynamodb.schema import PK_TMCACHE_PREFIX, SK_TMCACHE_RESULT

    item = table.get_item(pk=f"{PK_TMCACHE_PREFIX}{query_hash}", sk=SK_TMCACHE_RESULT)
    if item is None:
        return None
    if item.get("ttl") and int(item["ttl"]) < int(time.time()):
        return None
    data = json.loads(item["data"])
    return [
        MetricDataPoint(
            metric_name=d["metric_name"],
            dimensions=d["dimensions"],
            values=d["values"],
        )
        for d in data
    ]
